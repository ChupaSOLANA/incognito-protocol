#!/usr/bin/env python3
"""Fix smart quotes in app.py"""

with open('services/api/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace smart quotes with regular ASCII quotes
replacements = [
    ('\u2018', "'"),  # Left single quotation mark
    ('\u2019', "'"),  # Right single quotation mark
    ('\u201c', '"'),  # Left double quotation mark
    ('\u201d', '"'),  # Right double quotation mark
]

for old, new in replacements:
    content = content.replace(old, new)

with open('services/api/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Replaced smart quotes with ASCII quotes')
