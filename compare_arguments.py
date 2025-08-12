#!/usr/bin/env python3
"""
Validation script to compare original main.py with refactored version.

This script helps validate that the refactoring maintains functionality
while improving code organization and removing redundancies.
"""

import argparse
import re
import sys
from typing import Dict, List, Set, Tuple


def extract_arguments_from_file(filepath: str) -> Dict[str, dict]:
    """Extract all command-line arguments from a Python file"""
    arguments = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Pattern to match add_argument calls
        pattern = r"add_argument\(\s*['\"]--([^'\"]+)['\"].*?\)"
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            arg_name = match.group(1)
            arg_block = match.group(0)
            
            # Extract argument details
            arg_info = {
                'name': arg_name,
                'block': arg_block,
                'type': extract_type(arg_block),
                'default': extract_default(arg_block),
                'help': extract_help(arg_block),
                'action': extract_action(arg_block)
            }
            
            arguments[arg_name] = arg_info
            
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return {}
    
    return arguments


def extract_type(arg_block: str) -> str:
    """Extract argument type from add_argument block"""
    type_match = re.search(r"type\s*=\s*(\w+)", arg_block)
    return type_match.group(1) if type_match else "str"


def extract_default(arg_block: str) -> str:
    """Extract default value from add_argument block"""
    default_match = re.search(r"default\s*=\s*([^,\)]+)", arg_block)
    return default_match.group(1).strip() if default_match else "None"


def extract_help(arg_block: str) -> str:
    """Extract help text from add_argument block"""
    help_match = re.search(r"help\s*=\s*['\"]([^'\"]*)['\"]", arg_block)
    return help_match.group(1) if help_match else ""


def extract_action(arg_block: str) -> str:
    """Extract action from add_argument block"""
    action_match = re.search(r"action\s*=\s*['\"]([^'\"]+)['\"]", arg_block)
    return action_match.group(1) if action_match else ""


def compare_arguments(original_args: Dict, refactored_args: Dict) -> None:
    """Compare arguments between original and refactored versions"""
    
    original_names = set(original_args.keys())
    refactored_names = set(refactored_args.keys())
    
    print("=== ARGUMENT COMPARISON REPORT ===\n")
    
    # Arguments only in original (potentially removed)
    only_original = original_names - refactored_names
    if only_original:
        print(f"REMOVED ARGUMENTS ({len(only_original)}):")
        for arg in sorted(only_original):
            help_text = original_args[arg]['help'] or "No help"
            print(f"  --{arg}: {help_text}")
        print()
    
    # Arguments only in refactored (newly added)
    only_refactored = refactored_names - original_names
    if only_refactored:
        print(f"NEW ARGUMENTS ({len(only_refactored)}):")
        for arg in sorted(only_refactored):
            help_text = refactored_args[arg]['help'] or "No help"
            print(f"  --{arg}: {help_text}")
        print()
    
    # Common arguments (check for changes)
    common_args = original_names & refactored_names
    changed_args = []
    
    for arg in common_args:
        orig = original_args[arg]
        refact = refactored_args[arg]
        
        changes = []
        if orig['type'] != refact['type']:
            changes.append(f"type: {orig['type']} -> {refact['type']}")
        if orig['default'] != refact['default']:
            changes.append(f"default: {orig['default']} -> {refact['default']}")
        if orig['action'] != refact['action']:
            changes.append(f"action: {orig['action']} -> {refact['action']}")
        
        if changes:
            changed_args.append((arg, changes))
    
    if changed_args:
        print(f"MODIFIED ARGUMENTS ({len(changed_args)}):")
        for arg, changes in changed_args:
            print(f"  --{arg}:")
            for change in changes:
                print(f"    {change}")
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"Original arguments: {len(original_names)}")
    print(f"Refactored arguments: {len(refactored_names)}")
    print(f"Removed: {len(only_original)}")
    print(f"Added: {len(only_refactored)}")
    print(f"Modified: {len(changed_args)}")
    print(f"Unchanged: {len(common_args) - len(changed_args)}")


def find_potential_redundancies(args: Dict) -> List[Tuple[str, str, str]]:
    """Find potentially redundant arguments"""
    redundancies = []
    
    # Group arguments by similar names/purposes
    batch_args = [name for name in args if 'batch' in name.lower()]
    memory_args = [name for name in args if 'memory' in name.lower() or 'mem' in name.lower()]
    threshold_args = [name for name in args if 'threshold' in name.lower() or '_th' in name.lower()]
    lr_args = [name for name in args if 'lr' in name.lower() or 'learning' in name.lower()]
    
    groups = [
        ("Batch-related", batch_args),
        ("Memory-related", memory_args), 
        ("Threshold-related", threshold_args),
        ("Learning rate-related", lr_args)
    ]
    
    for group_name, group_args in groups:
        if len(group_args) > 1:
            for arg in group_args:
                help_text = args[arg]['help'] or "No help"
                redundancies.append((group_name, arg, help_text))
    
    return redundancies


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description='Compare original and refactored main.py')
    parser.add_argument('--original', default='main.py', 
                       help='Path to original main.py')
    parser.add_argument('--refactored', default='main_refactored.py',
                       help='Path to refactored main.py')
    parser.add_argument('--show-redundancies', action='store_true',
                       help='Show potential redundancies in original')
    
    args = parser.parse_args()
    
    print("Extracting arguments from files...")
    original_args = extract_arguments_from_file(args.original)
    refactored_args = extract_arguments_from_file(args.refactored)
    
    if not original_args:
        print("Error: Could not extract arguments from original file")
        return 1
    
    if not refactored_args:
        print("Error: Could not extract arguments from refactored file")
        return 1
    
    # Compare arguments
    compare_arguments(original_args, refactored_args)
    
    # Show redundancies if requested
    if args.show_redundancies:
        print("\n=== POTENTIAL REDUNDANCIES IN ORIGINAL ===")
        redundancies = find_potential_redundancies(original_args)
        
        current_group = None
        for group, arg, help_text in redundancies:
            if group != current_group:
                print(f"\n{group}:")
                current_group = group
            print(f"  --{arg}: {help_text}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
