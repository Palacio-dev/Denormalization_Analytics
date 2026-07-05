"""
SQL Model Evaluation Script
This script compares a relational SQL model with its denormalized version
using BLEU, ROUGE, and METEOR metrics.
"""

import re
import nltk
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple
import argparse
import numpy as np
from scipy.optimize import linear_sum_assignment


class SQLParser:
    """Parser for SQL CREATE TABLE statements"""

    NUMERIC_TYPES = ('INT', 'FLOAT', 'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE', 'MONEY', 'BIGINT', 'SMALLINT', 'SERIAL', 
                     'INTEGER', 'DOUBLE PRECISION', 'SMALLSERIAL', 'BIGSERIAL')
    TEXT_TYPES = ('CHAR', 'VARCHAR', 'TEXT', 'STRING')

    @staticmethod
    def is_numeric_type(data_type: str) -> bool:
        return any(nt in data_type.upper() for nt in SQLParser.NUMERIC_TYPES)

    @staticmethod
    def is_text_type(data_type: str) -> bool:
        return any(tt in data_type.upper() for tt in SQLParser.TEXT_TYPES)

    @staticmethod
    def remove_sql_comments(text: str) -> str:
        """
        Remove SQL comments (both -- and /* */ style) from text.
        Handles inline comments and full-line comments.
        """
        # Remove -- style comments (from -- to end of line)
        text = re.sub(r'--.*?$', '', text, flags=re.MULTILINE)
        # Remove /* */ style comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    @staticmethod
    def extract_fk_references(line: str) -> List[Dict]:
        """
        Single source of truth for foreign-key detection on one logical line
        (a constraint line OR a column-definition line). Handles both:
          - table-level:  FOREIGN KEY (col1, col2) REFERENCES other_table
          - inline:       col_name TYPE ... REFERENCES other_table(...)
        Returns a list of {'local_column': str, 'target_table': str or None}.
        Previously this logic was reimplemented three times (parse_file's inline
        scan, get_table_role's constraint scan, OLAPAvaluator._extract_fk_references)
        with three slightly different regexes that could disagree on the same input.
        """
        line_upper = line.upper()
        refs = []

        if 'FOREIGN KEY' in line_upper:
            match = re.search(
                r'FOREIGN KEY\s*\((.*?)\)\s*(?:REFERENCES\s*([^\s\(]+))?',
                line, re.IGNORECASE
            )
            if match:
                cols = [c.strip().strip('"\'') for c in match.group(1).split(',')]
                target = match.group(2).strip() if match.group(2) else None
                for col in cols:
                    refs.append({'local_column': col, 'target_table': target})
            return refs

        if ' REFERENCES ' in line_upper:
            # Inline column-level FK: the column name is the first token on the line.
            local_col = line.split()[0].strip('"\'')
            target_match = re.search(r'REFERENCES\s*([^\s\(]+)', line, re.IGNORECASE)
            target = target_match.group(1).strip() if target_match else None
            refs.append({'local_column': local_col, 'target_table': target})

        return refs

    @staticmethod
    def parse_file(filepath: str) -> Dict[str, Dict]:
        """
        Parse SQL file and extract table definitions
        Returns: Dict with table_name -> {columns: [...], constraints: [...]}
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Remove all SQL comments from content first
        content = SQLParser.remove_sql_comments(content)
        
        tables = {}
        table_pattern_with_name = r'CREATE TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:(\w+)\.)?(?:"([^"]+)"|\'([^\']+)\'|(\w+))\s*\((.*?)\);'
        matches_with_names = re.findall(table_pattern_with_name, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches_with_names:
            table_name = match[1] or match[2] or match[3]
            table_body = match[4].strip()
            columns = []
            constraints = []
            fk_columns = []
            fk_references = []
            
            # Split by commas, but respect parentheses
            # This handles cases like: PRIMARY KEY (col1, col2)
            lines = []
            current_line = []
            paren_depth = 0
            
            for char in table_body + ',':  # Add trailing comma to process last item
                if char == '(':
                    paren_depth += 1
                    current_line.append(char)
                elif char == ')':
                    paren_depth -= 1
                    current_line.append(char)
                elif char == ',' and paren_depth == 0:
                    # Only split on commas outside parentheses
                    line = ''.join(current_line).strip()
                    if line:
                        lines.append(line)
                    current_line = []
                else:
                    current_line.append(char)
            
            for line in lines:
                # Skip lines that are just closing parenthesis or other noise
                if line in (')', ');', '(', '()', ',') or line.startswith(')'):
                    continue
                line_upper = line.upper()
                if (line_upper.startswith('PRIMARY KEY') or
                    line_upper.startswith('FOREIGN KEY') or
                    line_upper.startswith('CONSTRAINT') or   
                    line_upper.startswith('UNIQUE') or        
                    line_upper.startswith('CHECK')):          
                    constraints.append(line)
                    refs = SQLParser.extract_fk_references(line)
                    fk_references.extend(refs)
                    fk_columns.extend(r['local_column'] for r in refs)
                else:
                    # It's a column definition
                    # Additional validation: a column definition should have at least a name and type
                    # Skip if it looks like a fragment or noise
                    if len(line.split()) >= 2:
                        columns.append(line)
                        refs = SQLParser.extract_fk_references(line)
                        fk_references.extend(refs)
                        fk_columns.extend(r['local_column'] for r in refs)
                
            
            tables[table_name] = {
                'columns': columns,
                'constraints': constraints,
                'fk_blacklist': list(set(fk_columns)),  # Unique list of columns involved in foreign keys
                'fk_references': fk_references  # Structured: [{'local_column', 'target_table'}, ...]
            }
        
        # --- END OF CREATE TABLE LOOP ---
        
        # Now that all tables are created, sweep for external ALTER TABLE Foreign Keys
        alter_pattern = (
            r'ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:(?:\w+)\.)?'
            r'(?:"([^"]+)"|\'([^\']+)\'|(\w+))[^;]+?'
            r'FOREIGN\s+KEY\s*\((.*?)\)\s*(?:REFERENCES\s*([^\s\(;]+))?'
        )
        alter_matches = re.findall(alter_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in alter_matches:
            # The regex returns a tuple of 5 groups.
            # match[0] = double-quoted name, match[1] = single-quoted name, match[2] = unquoted name
            # match[3] = the foreign key column(s), match[4] = referenced target table (optional)
            
            table_name = match[0] or match[1] or match[2]
            fk_str = match[3]
            target_table = match[4].strip() if match[4] else None
            
            # Check if this table actually exists in our parsed dictionary
            if table_name in tables:
                # Split by commas and clean up quotes/spaces to handle composite keys
                cols = [c.strip().strip('"\'') for c in fk_str.split(',')]
                
                # Add the new columns to the existing blacklist
                tables[table_name]['fk_blacklist'].extend(cols)
                tables[table_name]['fk_references'].extend(
                    {'local_column': c, 'target_table': target_table} for c in cols
                )
                
                # Remove duplicates just to keep the list clean
                tables[table_name]['fk_blacklist'] = list(set(tables[table_name]['fk_blacklist']))

        return tables # <-- This is the final return statement of parse_file
    
    @staticmethod
    def extract_column_info(column_def: str) -> Dict:
        """
        Extract column name, data type, and constraints from column definition
        Returns: {name: str, type: str, constraints: [str]}
        """
        column_def = column_def.strip()
        parts = column_def.split()
        if len(parts) < 2:
            return None
        
        name = parts[0]
        data_type = parts[1]
        constraints = []
        if len(parts) > 2:
            constraints = parts[2:]
        rest = ' '.join(parts[1:])
        
        type_match = re.match(r'(\w+(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?)', rest)
        
        if type_match:
            data_type = type_match.group(1)
            data_type = re.sub(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'(\1, \2)', data_type)
            data_type = re.sub(r'\(\s*(\d+)\s*\)', r'(\1)', data_type)

            raw_type = type_match.group(1)
            true_constraints_str = rest.replace(raw_type, '', 1).strip()
            constraints = true_constraints_str.split() if true_constraints_str else []
        else:
            data_type = rest.split()[0]  # Fallback to first word after name
            constraints = rest.split()[1:] 

        
        return {
            'name': name,
            'type': data_type.upper(),
            'constraints': constraints,
            'full_def': column_def
        }
    
    @staticmethod
    def get_table_role(table_info: Dict) -> str:
        """
        Analyzes a table's structure to determine its architectural role.
        Returns: 'JUNCTION_TABLE', 'FACT'

        Note: relies on table_info['fk_blacklist'], which is populated by
        parse_file via extract_fk_references (covers both table-level
        FOREIGN KEY(...) constraints and inline "col TYPE REFERENCES ..."
        definitions). Previously this method re-derived FK columns with its
        own regex that only matched the table-level form, silently missing
        inline FKs and disagreeing with the rest of the codebase.
        """
        columns = table_info.get('columns', [])
        fk_columns = set(table_info.get('fk_blacklist', []))
        fk_count = len(fk_columns)

        column_names = set()
        numeric_count = 0
        text_count = 0

        for col_def in columns:
            col_info = SQLParser.extract_column_info(col_def)
            if col_info:
                col_name = col_info['name']
                col_type = col_info['type']
                column_names.add(col_name)

                # If it's NOT a Foreign Key, analyze its data type
                if col_name not in fk_columns:
                    if SQLParser.is_numeric_type(col_type):
                        numeric_count += 1
                    else:
                        text_count += 1

        # --- CLASSIFICATION LOGIC ---

        # RULE A: Pure Junction (Everything is an FK, or mostly FKs with no real payload)
        if len(column_names) > 0 and column_names == fk_columns:
            return 'JUNCTION_TABLE'
        if fk_count >= 2 and len(columns) <= 3:
            return 'JUNCTION_TABLE'

        return 'FACT'
    
    @staticmethod
    def is_metadata_column(column_name: str) -> bool:
        """
        Heuristic to detect if a column is a standard audit/system metadata attribute.
        Matches common patterns like created_at, modifieddate, rowguid, etc.
        """
        # Clean the name and convert to lowercase for easy matching
        name = column_name.lower().strip('"\'')
        
        # 1. Check for standard replication/system IDs
        if name in ('rowguid', 'guid', 'sys_id'):
            return True
            
        # 2. Check for common Audit Log combinations
        audit_verbs = ['creat', 'update', 'modif', 'delete', 'insert']
        audit_suffixes = ['at', 'by', 'date', 'time']
        
        for verb in audit_verbs:
            for suffix in audit_suffixes:
                # Matches: created_at, modifieddate, update_time, deleted_by, etc.
                if verb in name and suffix in name:
                    return True
                    
        return False
    
    @staticmethod
    def iter_filtered_columns(tables: Dict, exclude_junction_tables: bool = True,
                               exclude_foreign_keys: bool = True, exclude_metadata: bool = True):
        """
        Single source of truth for the "which columns count as real business
        attributes" filter chain: skip junction tables, skip FK columns, skip
        audit/metadata columns. Previously this exact three-step chain was
        copy-pasted across get_all_attributes, pair_identifiers, and
        build_combined_attribute_text -- any fix to the filter rules (e.g. the
        modifieddate/rowguid exclusion) had to be made in three places by hand,
        and it would be easy for them to silently drift apart again.

        Yields (table_name, col_info, raw_column_def) for every column that
        survives the filters.
        """
        for table_name, table_info in tables.items():
            if exclude_junction_tables and SQLParser.get_table_role(table_info) == 'JUNCTION_TABLE':
                continue

            blacklist = table_info.get('fk_blacklist', []) if exclude_foreign_keys else []

            for column in table_info['columns']:
                col_info = SQLParser.extract_column_info(column)
                if not col_info:
                    continue
                if exclude_foreign_keys and col_info['name'] in blacklist:
                    continue
                if exclude_metadata and SQLParser.is_metadata_column(col_info['name']):
                    continue
                yield table_name, col_info, column

    @staticmethod
    def get_all_attributes(tables: Dict, exclude_junction_tables: bool = True, exclude_foreign_keys: bool = True, 
                           exclude_metadata: bool = True) -> List[str]:
        """
        Get all column definitions (raw strings) from all tables, after
        applying the standard junction/FK/metadata filter chain.
        """
        return [
            raw_column for _, _, raw_column in SQLParser.iter_filtered_columns(
                tables, exclude_junction_tables, exclude_foreign_keys, exclude_metadata
            )
        ]


class OLAPAvaluator:
    def __init__(self, denormalized_tables: Dict[str, Dict]):
        self.tables = denormalized_tables
        self.classifications = {}
        self.report = {
            "fact_tables": [],
            "dimension_tables": [],
            "warnings": [],
            "metrics": {}
        }

    def _fk_references(self, table_info: Dict) -> List[Dict]:
        """
        FK references for a table, derived from its already-parsed
        'fk_references' (built once by SQLParser.parse_file). Falls back to
        re-deriving from constraints for tables that didn't go through
        parse_file's structured path.
        """
        if 'fk_references' in table_info:
            return table_info['fk_references']
        refs = []
        for constraint in table_info.get('constraints', []):
            refs.extend(SQLParser.extract_fk_references(constraint))
        return refs

    def _parsed_columns(self, table_info: Dict) -> List[Dict]:
        """
        Converts the raw column-definition strings in table_info['columns']
        into structured dicts via SQLParser.extract_column_info.
        table_info['columns'] is always a list of strings like
        "customer_id INT NOT NULL" -- never pre-parsed dicts. Code here used
        to call col.get('name', '') directly on those strings, which raises
        AttributeError the moment any fact/dimension table has columns.
        """
        parsed = []
        for col_def in table_info.get('columns', []):
            col_info = SQLParser.extract_column_info(col_def)
            if col_info:
                parsed.append(col_info)
        return parsed

    def classify_tables(self):
        """Classify tables as FACT or DIMENSION based on FK count."""
        for table_name, table_info in self.tables.items():
            fks = self._fk_references(table_info)

            if len(fks) >= 2:
                self.classifications[table_name] = 'FACT'
                self.report['fact_tables'].append(table_name)
            else:
                self.classifications[table_name] = 'DIMENSION'
                self.report['dimension_tables'].append(table_name)


    def evaluate_one_big_table(self):
        """
        Run the specific checks for a One-Big-Table (OBT) denormalization approach.
        In a pure OBT, there should be exactly 1 main table (or isolated tables),
        0 Foreign Keys, and at least 1 Primary Key.
        """
        table_count = len(self.tables)

        # General Architecture Warning
        if table_count > 1:
            self.report['warnings'].append(
                f"OBT ARCHITECTURE WARNING: Found {table_count} tables. A true One-Big-Table approach typically collapses data into exactly 1 table."
            )

        for table_name, table_info in self.tables.items():
            constraints = table_info.get('constraints', [])
            columns = self._parsed_columns(table_info)

            fks = self._fk_references(table_info)

            # Count PKs once: table-level constraints OR column-level "PRIMARY KEY"
            # token, never both for the same key (previous version double-counted
            # whenever a PK appeared in constraints AND tried to re-read it off
            # raw column strings, which also crashed on col.get(...)).
            pk_count = sum(1 for c in constraints if 'PRIMARY KEY' in c.upper())
            if pk_count == 0:
                pk_count = sum(
                    1 for col in columns
                    if any('PRIMARY' in c.upper() for c in col['constraints'])
                )

            # Check 1: The "Zero Link" Rule (No Foreign Keys)
            if len(fks) > 0:
                self.report['warnings'].append(
                    f"OBT FK VIOLATION: Table '{table_name}' contains {len(fks)} Foreign Key(s). A pure One-Big-Table must have 0 relationships."
                )

            # Check 2: The Identity Rule (Must have a Primary Key)
            if pk_count == 0:
                self.report['warnings'].append(
                    f"OBT PK WARNING: Table '{table_name}' lacks a Primary Key. Even deeply flattened tables require a unique identifier."
                )


    def evaluate_galaxy_schema(self):
        """Ensure no Fact table has a Foreign Key pointing to another Fact table."""
        for table_name in self.report['fact_tables']:
            fks = self._fk_references(self.tables[table_name])
            for fk in fks:
                target = fk.get('target_table')
                if not target:
                    continue
                target_clean = re.sub(r'["\[\]]', '', target)
                target_clean = target_clean.split('.')[-1] if '.' in target_clean else target_clean

                # Search our classifications (ignoring schema prefixes if needed)
                for class_table_name, classification in self.classifications.items():
                    class_table_clean = class_table_name.split('.')[-1] if '.' in class_table_name else class_table_name

                    if class_table_clean == target_clean and classification == 'FACT':
                        self.report['warnings'].append(
                            f"GALAXY VIOLATION: Fact table '{table_name}' has an FK pointing to Fact '{target}'. Facts must only join to Dimensions."
                        )


    def evaluate_fact_tables(self):
        """Run the Kimball-derived checks for Fact tables."""
        for table_name in self.report['fact_tables']:
            table_info = self.tables[table_name]
            columns = self._parsed_columns(table_info)
            fk_columns = {fk['local_column'] for fk in self._fk_references(table_info)}

            numeric_count = 0
            text_count = 0

            for col in columns:
                col_name = col['name']
                col_type = col['type']
                col_constraints_upper = ' '.join(col['constraints']).upper()
                is_nullable = 'NOT NULL' not in col_constraints_upper

                # Check: NOT NULL Foreign Key Check.
                # Kimball: "nulls must be avoided in the fact table's foreign
                # keys because these nulls would automatically cause a
                # referential integrity violation" (Ch.2, Nulls in Fact Tables).
                if col_name in fk_columns and is_nullable:
                    self.report['warnings'].append(
                        f"FACT FK VIOLATION: Foreign Key '{col_name}' in Fact '{table_name}' allows NULL values."
                    )

                # Payload counting (ignore keys for payload analysis)
                if col_name not in fk_columns and 'PRIMARY' not in col_constraints_upper:
                    if SQLParser.is_numeric_type(col_type):
                        numeric_count += 1
                    else:
                        text_count += 1

            # Payload check: a fact table dominated by text columns is a smell --
            # Kimball facts are "almost always numeric" (Ch.2, Facts for Measurements).
            self.report['metrics'][table_name] = {
                "numeric_payload_cols": numeric_count,
                "text_payload_cols": text_count
            }
            if text_count > numeric_count:
                self.report['warnings'].append(
                    f"PAYLOAD WARNING: Fact table '{table_name}' contains more text attributes ({text_count}) than numeric measures ({numeric_count})."
                )

    def evaluate_dimension_tables(self):
        """Run the Kimball-derived checks for Dimension tables."""
        for table_name in self.report['dimension_tables']:
            table_info = self.tables[table_name]
            columns = self._parsed_columns(table_info)
            constraints = table_info.get('constraints', [])

            # Snowflake Audit: a dimension with outgoing FKs to other attribute
            # tables hasn't been flattened (Ch.2, Snowflaked Dimensions).
            fks = self._fk_references(table_info)
            if len(fks) > 0:
                self.report['warnings'].append(
                    f"SNOWFLAKE VIOLATION: Dimension '{table_name}' contains {len(fks)} Foreign Key(s). It is not fully flattened."
                )

            pk_count = 0
            numeric_count = 0
            text_count = 0

            for col in columns:
                col_name = col['name']
                col_type = col['type']
                col_constraints = ' '.join(col['constraints']).upper()

                is_pk_col = 'PRIMARY KEY' in col_constraints

                if is_pk_col:
                    pk_count += 1
                else:
                    # No NULL Attributes (Ch.2, Null Attributes in Dimensions --
                    # nulls should be replaced with an explicit "Unknown"/"N/A" row).
                    if 'NOT NULL' not in col_constraints:
                        self.report['warnings'].append(
                            f"DIMENSION NULL WARNING: Attribute '{col_name}' in '{table_name}' allows NULLs. Consider demanding default values."
                        )

                    if SQLParser.is_numeric_type(col_type):
                        numeric_count += 1
                    else:
                        text_count += 1

            # Single Primary Key check (Ch.2, Dimension Surrogate Keys: "every
            # dimension table has a single primary key column").
            if pk_count == 0:
                pk_in_constraints = sum(1 for c in constraints if 'PRIMARY KEY' in c.upper())
                if pk_in_constraints != 1:
                    self.report['warnings'].append(
                        f"PK VIOLATION: Dimension '{table_name}' does not have exactly 1 Primary Key."
                    )
            elif pk_count > 1:
                self.report['warnings'].append(
                    f"PK VIOLATION: Dimension '{table_name}' has a composite Primary Key ({pk_count} cols). It should use a single Surrogate Key."
                )

            self.report['metrics'][table_name] = {
                "numeric_payload_cols": numeric_count,
                "text_payload_cols": text_count
            }

    def run_full_audit(self):
        """
        Executes the complete architectural audit for OLAP structures.
        Dynamically adjusts checks based on Star vs. Galaxy detection.
        """
        # 1. Categorize all tables into Facts and Dimensions first
        self.classify_tables()

        # 2. Extract the count of Fact tables to determine the exact OLAP subtype
        fact_count = len(self.report['fact_tables'])

        # 3. Conditional Galaxy Schema Audit
        if fact_count > 1:
            self.report['warnings'].append(
                f"[i] INFO: Detected {fact_count} Fact tables. Evaluating as a GALAXY SCHEMA."
            )
            # Only run the cross-fact validation if it's actually a Galaxy
            self.evaluate_galaxy_schema()
        elif fact_count == 1:
            self.report['warnings'].append(
                "[i] INFO: Detected exactly 1 Fact table. Evaluating as a standard STAR SCHEMA."
            )
        else:
            self.report['warnings'].append(
                "[!] CRITICAL: OLAP schema detected, but 0 Fact tables found. Evaluation may be compromised."
            )

        # 4. Proceed with standard Kimball validations regardless of fact_count,
        # so dimension-only issues are still reported even in the 0-fact case.
        self.evaluate_fact_tables()
        self.evaluate_dimension_tables()

        return self.report

class ModelComparator:
    """Compare relational and denormalized models"""
    
    def __init__(self, relational_file: str, denormalized_file: str):
        self.relational_tables = SQLParser.parse_file(relational_file)
        self.denormalized_tables = SQLParser.parse_file(denormalized_file)
        
        self.relational_attrs = SQLParser.get_all_attributes(self.relational_tables)
        self.denormalized_attrs = SQLParser.get_all_attributes(self.denormalized_tables)
    
    def calculate_completeness(self) -> float:
        """
        Calculate completeness ratio: 
        num_attributes(relational) / num_attributes(denormalized)
        Ideal value is 1.0
        """
        rel_count = len(self.relational_attrs)
        denorm_count = len(self.denormalized_attrs)
        
        if rel_count == 0:
            return 0.0

        return denorm_count / rel_count
    
    
    def pair_identifiers(self, exclude_foreign_keys: bool = True, exclude_metadata: bool = True) -> Tuple[List[Tuple[Dict, Dict]], float]:
        """
        Pair equivalent identifiers between relational and denormalized models.
        Uses the following matching strategy:
        - Best Levenshtein distance (for remaining unpaired attributes)
    
        Returns:
        A tuple containing:
        - List of (relational_attr_dict, denormalized_attr_dict) tuples
        - The average Levenshtein distance of the matched pairs (float)
        """
        rel_columns_list = []
        for table_name, col_info, _ in SQLParser.iter_filtered_columns(
            self.relational_tables, exclude_junction_tables=True,
            exclude_foreign_keys=exclude_foreign_keys, exclude_metadata=exclude_metadata
        ):
            col_info = dict(col_info)  # avoid mutating the cached dict with 'table'
            col_info['table'] = table_name
            rel_columns_list.append(col_info)

        denorm_columns_list = []
        for table_name, col_info, _ in SQLParser.iter_filtered_columns(
            self.denormalized_tables, exclude_junction_tables=True,
            exclude_foreign_keys=exclude_foreign_keys, exclude_metadata=exclude_metadata
        ):
            col_info = dict(col_info)
            col_info['table'] = table_name
            denorm_columns_list.append(col_info)
        
        # Build cost matrix: rows = relational columns, cols = denormalized columns
        cost_matrix = np.zeros((len(rel_columns_list), len(denorm_columns_list)))
        for i, rel_info in enumerate(rel_columns_list):
            for j, denorm_info in enumerate(denorm_columns_list):
                cost_matrix[i][j] = Levenshtein.distance(
                    rel_info['name'], denorm_info['name']
                )

        # Hungarian algorithm finds the globally optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        pairs = []
        total_distance = 0.0  # 2. Initialize a running total for the distance

        for i, j in zip(row_indices, col_indices):
            pairs.append((rel_columns_list[i], denorm_columns_list[j]))
            
            # 3. Add the optimal pair's distance to our total
            total_distance += cost_matrix[i][j]

        # 4. Calculate the average safely (avoiding division by zero)
        num_pairs = len(pairs)
        average_distance = (total_distance / num_pairs) if num_pairs > 0 else 0.0

        # 5. Return both the pairs list and the calculated average
        return pairs, average_distance
    

    def calculate_bleu(self, reference: str, candidate: str, weights=(1.0,)) -> float:
        """Calculate BLEU score"""
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        try:
            score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        except:
            score = 0.0
        
        return score
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict:
        """Calculate ROUGE scores (ROUGE-1)"""
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
        }
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        try:
            # Ensure NLTK resources are available
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)
        
        try:
            score = meteor_score([ref_tokens], cand_tokens)
        except:
            score = 0.0
        
        return score
    
    def evaluate_correctness(self) -> Dict:
        """
        Evaluate correctness by checking:
        1. Data types preservation
        """
        results = {
            'type_preservation': [],
            'issues': []
        }
        
        pairs, avg_levenshtein = self.pair_identifiers()
        
        for rel_info, denorm_info in pairs:
            if not rel_info or not denorm_info:
                continue
            
            # Check data type preservation
            type_match = rel_info['type'] == denorm_info['type']
            results['type_preservation'].append({
                'relational': f"{rel_info['name']} {rel_info['type']}",
                'denormalized': f"{denorm_info['name']} {denorm_info['type']}",
                'match': type_match
            })
            
            if not type_match:
                results['issues'].append(
                    f"Type mismatch: {rel_info['name']} ({rel_info['type']}) -> "
                    f"{denorm_info['name']} ({denorm_info['type']})"
                )
        
        # Calculate percentages
        if results['type_preservation']:
            type_matches = sum(1 for x in results['type_preservation'] if x['match'])
            results['type_preservation_rate'] = type_matches / len(results['type_preservation'])
        else:
            results['type_preservation_rate'] = 0.0
        
        results['average_levenshtein'] = avg_levenshtein
        return results
    
    def build_combined_attribute_text(self, tables: Dict, exclude_junction_tables: bool = True, exclude_foreign_keys: bool = True, exclude_metadata: bool = True) -> str:
        """
        Build combined text from all column names in tables.
        Extracts all column names, sorts them alphabetically, and joins as space-separated string.
        """
        column_names = [
            col_info['name'] for _, col_info, _ in SQLParser.iter_filtered_columns(
                tables, exclude_junction_tables, exclude_foreign_keys, exclude_metadata
            )
        ]

        # Sort alphabetically for reproducible comparison
        column_names.sort()

        # Join as space-separated string
        return ' '.join(column_names)
    
    def calculate_structural_metrics(self) -> Dict:
        """
        Calculate BLEU, ROUGE, and METEOR scores on combined, alphabetically-sorted attribute names.
        This measures structural/naming similarity between relational and denormalized models.
        
        Returns:
            Dictionary with metric scores and combined texts
        """
        # Build combined texts from both models
        rel_text = self.build_combined_attribute_text(self.relational_tables)
        denorm_text = self.build_combined_attribute_text(self.denormalized_tables)
        
        results = {
            'relational_text': rel_text,
            'denormalized_text': denorm_text,
        }
        
        # Calculate metrics: reference = relational, candidate = denormalized
        bleu = self.calculate_bleu(rel_text, denorm_text)
        rouge = self.calculate_rouge(rel_text, denorm_text)
        meteor = self.calculate_meteor(rel_text, denorm_text)
        
        results['bleu'] = bleu
        results['rouge1'] = rouge['rouge1']
        results['meteor'] = meteor
        
        return results


    def detect_strategy(self) -> str:
        """
        Determines the denormalization strategy used by the LLM based on
        table counts and architectural roles.
        """
        # 1. Check for One-Big-Table (OBT)
        # If the LLM squashed everything into exactly 1 table.
        if len(self.denormalized_tables) <= 1:
            return 'OBT'

        # 2. Classify tables as FACT/DIMENSION using the same logic the OLAP
        # auditor uses, so detect_strategy and run_full_audit never disagree
        # about what counts as a fact table. (Previously this called
        # SQLParser.classify_table, which does not exist anywhere in this
        # module and raised AttributeError on every multi-table schema.)
        olap_eval = OLAPAvaluator(self.denormalized_tables)
        olap_eval.classify_tables()

        # 3. Check for Dimensional Modeling (Star/Galaxy)
        # If at least one table acts as a central hub with multiple FKs.
        if olap_eval.report['fact_tables']:
            return 'OLAP'

        # 4. Fallback for undefined/messy architectures
        # Multiple tables exist, but no clear Fact table bridges them.
        return 'Generic'
    
    def generate_report(self, output_file: str = None):
        """Generate comprehensive evaluation report"""
        # Build report content
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("SQL MODEL DENORMALIZATION EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # 1. Model Overview
        report_lines.append("\n1. MODEL OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Relational Model:")
        report_lines.append(f"  - Tables: {len(self.relational_tables)}")
        report_lines.append(f"  - Total Attributes: {len(self.relational_attrs)}")
        junction_tables = []
        for table_name, table_info in self.relational_tables.items():
            is_junction = SQLParser.get_table_role(table_info) == 'JUNCTION_TABLE'
            marker = " (junction table - excluded from attribute analysis)" if is_junction else ""
            report_lines.append(f"    * {table_name}: {len(table_info['columns'])} columns{marker}")
            if is_junction:
                junction_tables.append(table_name)

        if junction_tables:
            report_lines.append(f"\n  Note: {len(junction_tables)} junction table(s) detected and excluded from attribute analysis:")
            for jt in junction_tables:
                report_lines.append(f"    - {jt}")
        
        report_lines.append(f"\nDenormalized Model:")
        report_lines.append(f"  - Tables: {len(self.denormalized_tables)}")
        report_lines.append(f"  - Total Attributes: {len(self.denormalized_attrs)}")
        for table_name, table_info in self.denormalized_tables.items():
            report_lines.append(f"    * {table_name}: {len(table_info['columns'])} columns")
        
        # 2. Completeness
        report_lines.append("\n2. COMPLETENESS ANALYSIS")
        report_lines.append("-" * 80)
        completeness = self.calculate_completeness()
        if completeness == 1.0:
            report_lines.append("✓ IDEAL: No information was added or lost")
        elif completeness > 1.0:
            report_lines.append(f"⚠ WARNING: Denormalized model has MORE attributes ({len(self.relational_attrs)} vs {len(self.denormalized_attrs)})")
            report_lines.append("  New information may have been added in the transition")
        else:
            report_lines.append(f"⚠ WARNING: Normalized model has MORE attributes ({len(self.denormalized_attrs)} vs {len(self.relational_attrs)})")
            report_lines.append("  Some information may have been lost in the transition")
        
        # 3. Correctness
        report_lines.append("\n3. CORRECTNESS EVALUATION")
        report_lines.append("-" * 80)
        correctness = self.evaluate_correctness()
        structural_metrics = self.calculate_structural_metrics()

        if correctness['issues']:
            report_lines.append("\n⚠ Issues Found:")
            for issue in correctness['issues']:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("\n✓ No type mismatches found")

        report_lines.append("-" * 80)
        
        report_lines.append(f"Completeness Ratio:        {completeness:.2f}")
        report_lines.append(f"Data Type Preservation:    {correctness['type_preservation_rate']:.2f}")
        report_lines.append(f"Avg Levenshtein Distance:  {correctness['average_levenshtein']:.2f}")
        report_lines.append(f"BLEU Score:                {structural_metrics['bleu']:.2f}")
        report_lines.append(f"ROUGE-1:                   {structural_metrics['rouge1']:.2f}")
        report_lines.append(f"METEOR Score:              {structural_metrics['meteor']:.2f}")
        
        strategy = self.detect_strategy()
        if strategy == 'OLAP':
            report_lines.append("\n4. OLAP ARCHITECTURE (STAR/GALAXY)")
            report_lines.append("-" * 80)
            olap_eval = OLAPAvaluator(self.denormalized_tables)
            olap_report = olap_eval.run_full_audit()

            report_lines.append(f"Fact Tables Detected: {', '.join(olap_report['fact_tables']) or 'None'}")
            report_lines.append(f"Dimension Tables Detected: {', '.join(olap_report['dimension_tables']) or 'None'}")

            if olap_report['warnings']:
                report_lines.append("\n⚠ Architectural Violations/Warnings:")
                for warning in olap_report['warnings']:
                    report_lines.append(f"  - {warning}")
            else:
                report_lines.append("\n✓ Perfect Star Schema Architecture Detected! No violations.")

            report_lines.append("\nPayload Analysis (Text vs Numeric Metrics):")
            for table, metrics in olap_report['metrics'].items():
                report_lines.append(f"  * {table}: {metrics['numeric_payload_cols']} Numeric, {metrics['text_payload_cols']} Text")

        if strategy == 'OBT':
            report_lines.append("\n4. ONE-BIG-TABLE (OBT) ARCHITECTURE AUDIT")
            report_lines.append("-" * 80)
            obt_eval = OLAPAvaluator(self.denormalized_tables)
            obt_eval.evaluate_one_big_table()
            obt_report = obt_eval.report

            if obt_report['warnings']:
                report_lines.append("\n⚠ OBT Violations/Warnings:")
                for warning in obt_report['warnings']:
                    report_lines.append(f"  - {warning}")
            else:
                report_lines.append("\n✓ No OBT violations detected! The architecture resembles a true One-Big-Table approach.")

        report_lines.append("\n" + "=" * 80)

        report_lines.append("-" * 80)

        report_lines.append("Relational Model (Combined Text):")
        report_lines.append(f"  {structural_metrics['relational_text']}\n")
        
        report_lines.append("Denormalized Model (Combined Text):")
        report_lines.append(f"  {structural_metrics['denormalized_text']}\n")
        
        # 5. Attribute Pairing for Type Validation
        report_lines.append("\n5. ATTRIBUTE PAIRING FOR TYPE VALIDATION")
        report_lines.append("-" * 80)
        
        pairs, avg_levenshtein = self.pair_identifiers()
        report_lines.append(f"\nIdentified {len(pairs)} attribute pairs for type checking:\n")
        
        for i, (rel_info, denorm_info) in enumerate(pairs, 1):
            report_lines.append(f"Pair {i}:")
            report_lines.append(f"  Relational:    {rel_info['name']} {rel_info['type']}")
            report_lines.append(f"  Denormalized:  {denorm_info['name']} {denorm_info['type']}")
            report_lines.append("")
        
        report_lines.append("\n" + "=" * 80 + "\n")
        # Print to console
        report_text = "\n".join(report_lines)
        # print(report_text)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SQL model denormalization using NLP metrics'
    )
    parser.add_argument(
        'relational_file',
        help='Path to the relational SQL model file'
    )
    parser.add_argument(
        'denormalized_file',
        help='Path to the denormalized SQL model file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file for the report (optional)',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        comparator = ModelComparator(args.relational_file, args.denormalized_file)
        comparator.generate_report(args.output)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())