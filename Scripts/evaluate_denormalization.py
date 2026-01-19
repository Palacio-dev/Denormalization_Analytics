"""
SQL Model Evaluation Script
This script compares a relational SQL model with its denormalized version
using BLEU, ROUGE, and METEOR metrics.
"""

import re
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from typing import Dict, List, Tuple
import argparse


class SQLParser:
    """Parser for SQL CREATE TABLE statements"""
    
    @staticmethod
    def parse_file(filepath: str) -> Dict[str, Dict]:
        """
        Parse SQL file and extract table definitions
        Returns: Dict with table_name -> {columns: [...], constraints: [...]}
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        tables = {}
        # Find all CREATE TABLE statements
        table_pattern = r'CREATE TABLE\s+(\w+)\s*\((.*?)\);'
        matches = re.findall(table_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for table_name, table_body in matches:
            columns = []
            constraints = []
            
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
                line = line.strip()
                if not line:
                    continue
                    
                # Check if it's a constraint (PRIMARY KEY, FOREIGN KEY)
                if line.upper().startswith('PRIMARY KEY') or \
                   line.upper().startswith('FOREIGN KEY'):
                    constraints.append(line)
                else:
                    # It's a column definition
                    columns.append(line)
            
            tables[table_name] = {
                'columns': columns,
                'constraints': constraints
            }
        
        return tables
    
    @staticmethod
    def extract_column_info(column_def: str) -> Dict:
        """
        Extract column name, data type, and constraints from column definition
        Returns: {name: str, type: str, constraints: [str]}
        """
        parts = column_def.split()
        if len(parts) < 2:
            return None
        
        name = parts[0]
        data_type = parts[1]
        
        # Handle types with size like VARCHAR(4)
        if '(' in data_type and ')' in data_type:
            type_match = re.match(r'(\w+)\((\d+)\)', data_type)
            if type_match:
                data_type = f"{type_match.group(1)}({type_match.group(2)})"
        
        # Extract constraints (PRIMARY KEY, NOT NULL, etc.)
        constraints = []
        if len(parts) > 2:
            constraints = parts[2:]
        
        return {
            'name': name,
            'type': data_type,
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
        fk_columns = set()
        for constraint in constraints:
            if 'FOREIGN KEY' in constraint.upper():
                # Pattern: FOREIGN KEY (column_name) REFERENCES ...
                fk_match = re.search(r'FOREIGN KEY\s*\((\w+)\)', constraint, re.IGNORECASE)
                if fk_match:
                    fk_columns.add(fk_match.group(1))
        
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
    def get_all_attributes(tables: Dict, exclude_junction_tables: bool = True) -> List[str]:
        """
        Get all column definitions from all tables
        
        Args:
            tables: Dictionary of tables
            exclude_junction_tables: If True, skip many-to-many junction tables
        """
        attributes = []
        for table_name, table_info in tables.items():
            # Skip junction tables if requested
            if exclude_junction_tables and SQLParser.is_junction_table(table_info):
                continue
            
            for column in table_info['columns']:
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

        return denorm_count / rel_count
    
    
    def pair_identifiers(self) -> List[Tuple[str, str]]:
        """
        Pair equivalent identifiers between relational and denormalized models.
        Uses a multi-stage matching strategy:
        1. Exact name match
        2. Partial name match (substring)
        3. Type-only match (for remaining unpaired attributes)
        
        Returns list of (relational_attr, denormalized_attr) tuples
        """
        pairs = []
        
        # Extract column information (excluding junction tables)
        # Use list to preserve all columns (even with duplicate names from different tables)
        # Esse loop percorre o dicionário de tabelas relacionais, que é do tipo, chave : NOME DA TABELA, valor :
        # DICIONÁRIO COM INFORMAÇÕES DA TABELA, depois percorre as colunas da tabela extraindo as informações de 
        # cada coluna e colocando em uma lista
        # As informações da colunas estão em formato de dicionário com nome, tipo, constraints, etc. Ou seja, 
        # rel_columns_list é uma lista de dicionários com todas as coluna do modelo.
        rel_columns_list = []
        for table_name, table_info in self.relational_tables.items():
            # Skip junction tables in the relational model
            if SQLParser.is_junction_table(table_info):
                continue
                
            for col_def in table_info['columns']:
                col_info = SQLParser.extract_column_info(col_def)
                if col_info:
                    # Add table context to avoid name collisions
                    col_info['table'] = table_name
                    rel_columns_list.append(col_info)
        
        denorm_columns_list = []
        for table_name, table_info in self.denormalized_tables.items():
            for col_def in table_info['columns']:
                col_info = SQLParser.extract_column_info(col_def)
                if col_info:
                    col_info['table'] = table_name
                    denorm_columns_list.append(col_info)
        
        # Track which columns have been paired (by index)
        rel_paired = [False] * len(rel_columns_list)
        denorm_paired = [False] * len(denorm_columns_list)
        
        # Stage 1: Exact name match with same type
        for i, rel_info in enumerate(rel_columns_list):
            if rel_paired[i]:
                continue
            for j, denorm_info in enumerate(denorm_columns_list):
                if denorm_paired[j]:
                    continue
                # Exact name match AND same type
                if rel_info['name'] == denorm_info['name'] and rel_info['type'] == denorm_info['type']:
                    rel_comparison = f"{rel_info['name']} {rel_info['type']}"
                    denorm_comparison = f"{denorm_info['name']} {denorm_info['type']}"
                    pairs.append((rel_comparison, denorm_comparison))
                    rel_paired[i] = True
                    denorm_paired[j] = True
                    break
        
        # Stage 2: Partial name match (substring) with same type
        for i, rel_info in enumerate(rel_columns_list):
            if rel_paired[i]:
                continue
            for j, denorm_info in enumerate(denorm_columns_list):
                if denorm_paired[j]:
                    continue
                # Substring match AND same type
                if (rel_info['name'] in denorm_info['name'] or denorm_info['name'] in rel_info['name']) and \
                   rel_info['type'] == denorm_info['type']:
                    rel_comparison = f"{rel_info['name']} {rel_info['type']}"
                    denorm_comparison = f"{denorm_info['name']} {denorm_info['type']}"
                    pairs.append((rel_comparison, denorm_comparison))
                    rel_paired[i] = True
                    denorm_paired[j] = True
                    break
        
        # Stage 3: Type-only match for remaining unpaired attributes
        # This handles cases where names are completely different but types match
        for i, rel_info in enumerate(rel_columns_list):
            if rel_paired[i]:
                continue
            for j, denorm_info in enumerate(denorm_columns_list):
                if denorm_paired[j]:
                    continue
                # Same type only (as a fallback)
                if rel_info['type'] == denorm_info['type']:
                    rel_comparison = f"{rel_info['name']} {rel_info['type']}"
                    denorm_comparison = f"{denorm_info['name']} {denorm_info['type']}"
                    pairs.append((rel_comparison, denorm_comparison))
                    rel_paired[i] = True
                    denorm_paired[j] = True
                    break
        
        # Validation: Ensure we have the expected number of pairs
        rel_count = len(rel_columns_list)
        denorm_count = len(denorm_columns_list)
        expected_pairs = min(rel_count, denorm_count)
        
        if len(pairs) < expected_pairs:
            # This shouldn't happen if all stages work correctly, but log if it does
            unpaired_rel = [rel_columns_list[i]['name'] for i in range(len(rel_columns_list)) if not rel_paired[i]]
            unpaired_denorm = [denorm_columns_list[j]['name'] for j in range(len(denorm_columns_list)) if not denorm_paired[j]]
            print(f"\n⚠ Warning: Expected {expected_pairs} pairs but found {len(pairs)}")
            if unpaired_rel:
                print(f"  Unpaired relational attributes: {', '.join(unpaired_rel)}")
            if unpaired_denorm:
                print(f"  Unpaired denormalized attributes: {', '.join(unpaired_denorm)}")
        
        return pairs
    
    def tokenize_sql(self, sql_string: str) -> List[str]:
        """Tokenize SQL string for metric calculation"""
        # Remove extra spaces and split
        tokens = re.findall(r'\w+|\(|\)', sql_string)
        return tokens
    
    def calculate_bleu(self, reference: str, candidate: str, weights=(0.5, 0.5)) -> float:
        """Calculate BLEU score"""
        ref_tokens = self.tokenize_sql(reference)
        cand_tokens = self.tokenize_sql(candidate)
        
        try:
            score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        except:
            score = 0.0
        
        return score
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict:
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
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
        2. Semantic preservation of identifiers
        3. Cardinality preservation (basic check)
        """
        results = {
            'type_preservation': [],
            'semantic_preservation': [],
            'issues': []
        }
        
        pairs = self.pair_identifiers()
        
        for rel_def, denorm_def in pairs:
            rel_info = SQLParser.extract_column_info(rel_def)
            denorm_info = SQLParser.extract_column_info(denorm_def)
            
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
            
            # Check semantic preservation (same name or logical mapping)
            semantic_match = (rel_info['name'] == denorm_info['name'] or 
                            rel_info['name'] in denorm_info['name'] or
                            denorm_info['name'] in rel_info['name'])
            results['semantic_preservation'].append({
                'relational': rel_info['name'],
                'denormalized': denorm_info['name'],
                'match': semantic_match
            })
        
        # Calculate percentages
        if results['type_preservation']:
            type_matches = sum(1 for x in results['type_preservation'] if x['match'])
            results['type_preservation_rate'] = type_matches / len(results['type_preservation'])
        else:
            results['type_preservation_rate'] = 0.0
        
        if results['semantic_preservation']:
            semantic_matches = sum(1 for x in results['semantic_preservation'] if x['match'])
            results['semantic_preservation_rate'] = semantic_matches / len(results['semantic_preservation'])
        else:
            results['semantic_preservation_rate'] = 0.0
        
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
        
        # Identify junction tables
        junction_tables = []
        for table_name, table_info in self.relational_tables.items():
            is_junction = SQLParser.is_junction_table(table_info)
            marker = " (junction table - excluded from analysis)" if is_junction else ""
            report_lines.append(f"    * {table_name}: {len(table_info['columns'])} columns{marker}")
            if is_junction:
                junction_tables.append(table_name)
        
        if junction_tables:
            report_lines.append(f"\n  Note: {len(junction_tables)} junction table(s) detected and excluded from analysis:")
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
        report_lines.append(f"Completeness Ratio: {completeness:.4f}")
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
        
        report_lines.append(f"\nData Type Preservation: {correctness['type_preservation_rate']:.2%}")
        report_lines.append(f"Semantic Preservation: {correctness['semantic_preservation_rate']:.2%}")
        
        if correctness['issues']:
            report_lines.append("\n⚠ Issues Found:")
            for issue in correctness['issues']:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("\n✓ No type mismatches found")
        
        # 4. Metric-based Comparison
        report_lines.append("\n4. METRIC-BASED COMPARISON")
        report_lines.append("-" * 80)
        
        pairs = self.pair_identifiers()
        report_lines.append(f"\nIdentified {len(pairs)} attribute pairs for comparison:\n")
        
        all_bleu_scores = []
        all_rouge1_scores = []
        all_rouge2_scores = []
        all_rougeL_scores = []
        all_meteor_scores = []
        
        for i, (rel_attr, denorm_attr) in enumerate(pairs, 1):
            report_lines.append(f"Pair {i}:")
            report_lines.append(f"  Relational:    {rel_attr}")
            report_lines.append(f"  Denormalized:  {denorm_attr}")
            
            # Calculate metrics
            bleu = self.calculate_bleu(rel_attr, denorm_attr)
            rouge = self.calculate_rouge(rel_attr, denorm_attr)
            meteor = self.calculate_meteor(rel_attr, denorm_attr)
            
            all_bleu_scores.append(bleu)
            all_rouge1_scores.append(rouge['rouge1'])
            all_rouge2_scores.append(rouge['rouge2'])
            all_rougeL_scores.append(rouge['rougeL'])
            all_meteor_scores.append(meteor)
            
            report_lines.append(f"  BLEU:     {bleu:.4f}")
            report_lines.append(f"  ROUGE-1:  {rouge['rouge1']:.4f}")
            report_lines.append(f"  ROUGE-2:  {rouge['rouge2']:.4f}")
            report_lines.append(f"  ROUGE-L:  {rouge['rougeL']:.4f}")
            report_lines.append(f"  METEOR:   {meteor:.4f}")
            report_lines.append("")
        
        # 5. Overall Metrics Summary
        report_lines.append("\n5. OVERALL METRICS SUMMARY")
        report_lines.append("-" * 80)
        if all_bleu_scores:
            report_lines.append(f"Average BLEU Score:    {sum(all_bleu_scores)/len(all_bleu_scores):.4f}")
            report_lines.append(f"Average ROUGE-1:       {sum(all_rouge1_scores)/len(all_rouge1_scores):.4f}")
            report_lines.append(f"Average ROUGE-2:       {sum(all_rouge2_scores)/len(all_rouge2_scores):.4f}")
            report_lines.append(f"Average ROUGE-L:       {sum(all_rougeL_scores)/len(all_rougeL_scores):.4f}")
            report_lines.append(f"Average METEOR Score:  {sum(all_meteor_scores)/len(all_meteor_scores):.4f}")
        else:
            report_lines.append("No paired attributes found for comparison")
        
        report_lines.append("\n" + "=" * 80 + "\n")
        # Print to console
        report_text = "\n".join(report_lines)
        print(report_text)
        
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
