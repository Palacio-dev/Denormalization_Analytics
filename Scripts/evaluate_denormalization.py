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
                    if 'FOREIGN KEY' in line_upper:
                        fk_match = re.search(r'FOREIGN KEY\s*\((.*?)\)', line, re.IGNORECASE)
                        if fk_match:
                            cols = [c.strip().strip('"\'') for c in fk_match.group(1).split(',')]
                            fk_columns.extend(cols)
                else:
                    # It's a column definition
                    # Additional validation: a column definition should have at least a name and type
                    # Skip if it looks like a fragment or noise
                    if len(line.split()) >= 2:
                        columns.append(line)
                    
                        if ' REFERENCES ' in line_upper:
                            # The column name is ALWAYS the very first word on this line
                            inline_fk_name = line.split()[0]
                            
                            # Clean it up (remove quotes if they used "id_uf")
                            inline_fk_name = inline_fk_name.strip('"\'')
                            
                            # Add it to our blacklist!
                            fk_columns.append(inline_fk_name)
                
            
            tables[table_name] = {
                'columns': columns,
                'constraints': constraints,
                'fk_blacklist': list(set(fk_columns))  # Unique list of columns involved in foreign keys
            }
        
        # --- END OF CREATE TABLE LOOP ---
        
        # Now that all tables are created, sweep for external ALTER TABLE Foreign Keys
        alter_pattern = r'ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:(?:\w+)\.)?(?:"([^"]+)"|\'([^\']+)\'|(\w+))[^;]+?FOREIGN\s+KEY\s*\((.*?)\)'
        alter_matches = re.findall(alter_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for match in alter_matches:
            # The regex returns a tuple of 4 groups. 
            # match[0] = double-quoted name, match[1] = single-quoted name, match[2] = unquoted name
            # match[3] = The foreign key column(s)
            
            table_name = match[0] or match[1] or match[2]
            fk_str = match[3]
            
            # Check if this table actually exists in our parsed dictionary
            if table_name in tables:
                # Split by commas and clean up quotes/spaces to handle composite keys
                cols = [c.strip().strip('"\'') for c in fk_str.split(',')]
                
                # Add the new columns to the existing blacklist
                tables[table_name]['fk_blacklist'].extend(cols)
                
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
    def is_junction_table(table_info: Dict) -> bool:
        """
        Detect if a table is a many-to-many junction table.
        Criteria:
        1. Has FOREIGN KEY constraints
        2. All columns (or most) are part of foreign keys
        3. Typically has a composite primary key
        """
        columns = table_info['columns']
        constraints = table_info['constraints']
        
        # Check if there are FOREIGN KEY constraints
        has_foreign_keys = any('FOREIGN KEY' in constraint.upper() for constraint in constraints)
        
        if not has_foreign_keys:
            return False
        
        # Count how many FOREIGN KEY references exist
        fk_count = sum(1 for constraint in constraints if 'FOREIGN KEY' in constraint.upper())
        
        # If table has 2+ foreign keys and the number of columns is small (<=3),
        # it's likely a junction table
        # This covers cases like: user_id, entitlement_id, created_at (optional timestamp)
        if fk_count >= 2 and len(columns) <= 3:
            return True
        
        # Additional check: if ALL columns are referenced in FOREIGN KEYs
        # Extract column names from FOREIGN KEY constraints
        # Better extraction logic
        fk_columns = set()
        for constraint in constraints:
            if 'FOREIGN KEY' in constraint.upper():
                # Capture everything inside the parentheses
                fk_match = re.search(r'FOREIGN KEY\s*\((.*?)\)', constraint, re.IGNORECASE)
                if fk_match:
                    # Split by comma and strip spaces to handle composite keys!
                    cols = [c.strip() for c in fk_match.group(1).split(',')]
                    fk_columns.update(cols)
        
        # Get actual column names
        column_names = set()
        for col_def in columns:
            col_info = SQLParser.extract_column_info(col_def)
            if col_info:
                column_names.add(col_info['name'])
        
        # If all columns are foreign keys, it's definitely a junction table
        if column_names and fk_columns and column_names == fk_columns:
            return True
        
        return False
    
    @staticmethod
    def get_all_attributes(tables: Dict, exclude_junction_tables: bool = True, exclude_foreign_keys: bool = True) -> List[str]:
        """
        Get all column definitions from all tables
        
        Args:
            tables: Dictionary of tables
            exclude_junction_tables: If True, skip many-to-many junction tables
            exclude_foreign_keys: If True, skip foreign key columns based on the fk_blacklist identified during parsing
        """
        attributes = []
        for table_name, table_info in tables.items():
            # Skip junction tables if requested
            if exclude_junction_tables and SQLParser.is_junction_table(table_info):
                continue
            blacklist = table_info.get('fk_blacklist', [])
            for column in table_info['columns']:
                col_info = SQLParser.extract_column_info(column)
                if col_info:
                    if exclude_foreign_keys and col_info['name'] in blacklist:
                        continue
                    attributes.append(column)
        return attributes


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
    
    
    def pair_identifiers(self, exclude_foreign_keys: bool = True) -> Tuple[List[Tuple[Dict, Dict]], float]:
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
        for table_name, table_info in self.relational_tables.items():
            # Skip junction tables in the relational model
            if SQLParser.is_junction_table(table_info):
                continue
                
            # 1. Fetch the blacklist for this specific relational table
            blacklist = table_info.get('fk_blacklist', []) if exclude_foreign_keys else []
                
            for col_def in table_info['columns']:
                col_info = SQLParser.extract_column_info(col_def)
                if col_info:
                    # 2. Block the Foreign Keys from entering the matching pool!
                    if col_info['name'] in blacklist:
                        continue
                        
                    # Add table context to avoid name collisions
                    col_info['table'] = table_name
                    rel_columns_list.append(col_info)
        
        denorm_columns_list = []
        for table_name, table_info in self.denormalized_tables.items():
            blacklist = table_info.get('fk_blacklist', []) if exclude_foreign_keys else []
            for col_def in table_info['columns']:
                col_info = SQLParser.extract_column_info(col_def)
                if col_info:
                    if col_info['name'] in blacklist:
                        continue
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
    
    def build_combined_attribute_text(self, tables: Dict, exclude_junction_tables: bool = True) -> str:
        """
        Build combined text from all column names in tables.
        Extracts all column names, sorts them alphabetically, and joins as space-separated string.
        
        Args:
            tables: Dictionary of tables
            exclude_junction_tables: If True, skip many-to-many junction tables
            
        Returns:
            Space-separated string of sorted column names
        """
        column_names = []
        
        for table_name, table_info in tables.items():
            # Skip junction tables if requested
            if exclude_junction_tables and SQLParser.is_junction_table(table_info):
                continue
            
            for col_def in table_info['columns']:
                col_info = SQLParser.extract_column_info(col_def)
                if col_info:
                    column_names.append(col_info['name'])
        
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
        rel_text = self.build_combined_attribute_text(self.relational_tables, exclude_junction_tables=True)
        denorm_text = self.build_combined_attribute_text(self.denormalized_tables, exclude_junction_tables=True)
        
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
        for table_name, table_info in self.relational_tables.items():
            report_lines.append(f"    * {table_name}: {len(table_info['columns'])} columns")
        
        # # Identify junction tables
        # junction_tables = []
        # for table_name, table_info in self.relational_tables.items():
        #     is_junction = SQLParser.is_junction_table(table_info)
        #     marker = " (junction table - excluded from analysis)" if is_junction else ""
        #     report_lines.append(f"    * {table_name}: {len(table_info['columns'])} columns{marker}")
        #     if is_junction:
        #         junction_tables.append(table_name)
        
        # if junction_tables:
        #     report_lines.append(f"\n  Note: {len(junction_tables)} junction table(s) detected and excluded from analysis:")
        #     for jt in junction_tables:
        #         report_lines.append(f"    - {jt}")
        
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
    