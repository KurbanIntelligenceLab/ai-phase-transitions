import re

with open('c:/Users/User/Desktop/paper/main.tex', encoding='utf-8') as f:
    tex = f.read()

# Abstract word count
m = re.search(r'\\begin[{]abstract[}](.*?)\\end[{]abstract[}]', tex, re.DOTALL)
abstract = m.group(1).strip() if m else ''
clean = re.sub(r'\\[a-zA-Z]+[{][^}]*[}]', ' ', abstract)
clean = re.sub(r'\\[a-zA-Z]+', ' ', clean)
clean = re.sub(r'[$].*?[$]', 'MATH', clean)
clean = re.sub(r'\s+', ' ', clean).strip()
words = len(clean.split())
print(f'Abstract word count: {words} (limit: 200)')

# Keywords
m2 = re.search(r'\\begin[{]keyword[}](.*?)\\end[{]keyword[}]', tex, re.DOTALL)
if m2:
    raw = m2.group(1)
    kws = [k.strip().replace('\\sep','').strip() for k in raw.split('\\sep')]
    kws = [k for k in kws if k]
    print(f'Keywords: {len(kws)} (limit: 6) => {kws}')

# Line numbering
if '% \\linenumbers' in tex:
    print('Line numbering: DISABLED (uncomment for submission)')
elif '\\linenumbers' in tex:
    print('Line numbering: ENABLED')

# Data availability
if 'zenodo' in tex.lower():
    print('Data availability (Zenodo): PRESENT')
else:
    print('Data availability: MISSING')

# CRediT
if 'CRediT' in tex or 'Authorship Contribution' in tex:
    print('CRediT author contributions: PRESENT')

# Competing interests
if 'Competing Interest' in tex or 'competing' in tex.lower():
    print('Competing interests: PRESENT')

# Funding
if 'Acknowledgement' in tex or 'funding' in tex.lower() or 'Acknowledgment' in tex:
    print('Funding/Acknowledgements: check manually')
else:
    print('Funding/Acknowledgements: NOT FOUND')

# Reference style (author-year check)
if 'citep' in tex or 'citet' in tex:
    print('Reference style: author-year (citep/citet) - OK')

# Word count estimate (rough)
body = re.sub(r'\\begin[{].*?[}].*?\\end[{].*?[}]', '', tex, flags=re.DOTALL)
body_words = len(re.sub(r'\\[a-zA-Z]+[{][^}]*[}]|\\[a-zA-Z]+|[{}%]', ' ', body).split())
print(f'Estimated body word count: ~{body_words} (target: 5,000-10,000)')

# Check if methods come before results (IMRAD not required but methods first)
methods_pos = tex.find('Methodology')
results_pos = tex.find('Results')
if methods_pos < results_pos:
    print('Methods before Results: OK')

# Figures at 300 DPI
if '300' in tex:
    print('300 DPI figures: referenced in code')

# Table colors check
if 'cellcolor' in tex or 'rowcolor' in tex:
    print('WARNING: colored table cells detected (not allowed by QSS)')
else:
    print('Table colors: none detected - OK')
