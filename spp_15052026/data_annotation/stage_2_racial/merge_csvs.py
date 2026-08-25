import os, csv
from pathlib import Path

success_patterns = ('succes','success')
fail_patterns = ('fail',)

root = Path('.')

s_files = []
f_files = []
for p in root.rglob('*.csv'):
    name = p.name.lower()
    if name.startswith('merged_'):
        continue
    if any(pat in name for pat in success_patterns):
        s_files.append(str(p))
    elif any(pat in name for pat in fail_patterns):
        f_files.append(str(p))

s_files = sorted(set(s_files))
f_files = sorted(set(f_files))

print('Found', len(s_files), 'success files and', len(f_files), 'failed files')


def merge(files, outname):
    if not files:
        print('No files for', outname)
        return
    header = None
    rows = 0
    with open(outname, 'w', newline='') as outf:
        writer = None
        for fname in files:
            with open(fname, 'r', newline='') as inf:
                reader = csv.reader(inf)
                try:
                    h = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = h
                    writer = csv.writer(outf)
                    writer.writerow(header)
                else:
                    if h != header:
                        print('Warning: header mismatch in', fname)
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    print('Merged', len(files), '->', outname, 'rows written:', rows)

merge(s_files, 'merged_process_success.csv')
merge(f_files, 'merged_process_failed.csv')
