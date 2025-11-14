#!/usr/bin/env python3
"""Extract categories from existing_files.txt"""

import re
from collections import Counter


def extract_categories(filename):
    """Extract categories from a filename.

    Pattern: ...__(category1_category2_category3).extension
    """
    # Match everything between '__' and the file extension
    match = re.search(r'__([^.]+)\.\w+$', filename)
    if match:
        category_string = match.group(1)
        # Split by underscore to get individual categories
        categories = category_string.split('_')
        return [cat.strip().lower() for cat in categories if cat.strip()]
    return []


def merge_categories(category):
    """Merge similar categories and fix typos."""
    # Define merge rules
    merge_map = {
        'sandieo': 'sandiego',
        'mortage': 'mortgage',
        'scanscap': 'scansnap',
        'bank': 'banking',  # Merge bank into banking
    }

    return merge_map.get(category, category)


def main():
    all_categories = []

    # Read the existing_files.txt and process each line
    with open('existing_files.txt', 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and directory lines
            if not line or line.endswith(':') or 'total' in line or line.startswith('drwx'):
                continue

            # Extract filename from ls -l output
            parts = line.split()
            if len(parts) > 0:
                filename = parts[-1]
                categories = extract_categories(filename)
                # Apply merging
                categories = [merge_categories(cat) for cat in categories]
                all_categories.extend(categories)

    # Count occurrences
    category_counts = Counter(all_categories)

    # Get unique categories sorted by frequency
    unique_categories = sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))

    print(f"Total category tags: {len(all_categories)}")
    print(f"Unique categories found: {len(unique_categories)}")
    print("\nCategories (with counts):")
    for category, count in unique_categories:
        print(f"  {category}: {count}")

    # Write to categories.txt (just the category names, sorted)
    with open('categories.txt', 'w') as f:
        f.write("# Categories extracted from existing_files.txt\n")
        f.write(f"# Total unique categories: {len(unique_categories)}\n")
        f.write("# Categories with similar names have been merged\n")
        f.write("#\n")
        f.write("# Merged categories:\n")
        f.write("#   - sandieo → sandiego (typo fix)\n")
        f.write("#   - mortage → mortgage (typo fix)\n")
        f.write("#   - scanscap → scansnap (typo fix)\n")
        f.write("#   - bank → banking (merged)\n")
        f.write("\n")
        for category, count in unique_categories:
            f.write(f"{category}\n")

    print("\nCategories written to categories.txt")


if __name__ == '__main__':
    main()
