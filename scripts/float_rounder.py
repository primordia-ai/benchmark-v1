"""
Convert floats from a LaTeX table to a specific format.
"""
import re
import sys


def format_floats_in_latex_table(text):
    def repl(match):
        prefix = match.group(1)
        number = float(match.group(2))
        return f"{prefix}{number:.2f}"

    # Match: (prefix)(float) where prefix is space, &, or -; followed by float
    float_regex = re.compile(r'([ \t&-])(-?\d+\.\d+)(?=[ \t&\\]|$)')

    lines = text.splitlines()
    new_lines = [float_regex.sub(repl, line) for line in lines]

    return "\n".join(new_lines)


if __name__ == '__main__':
    # First argument is .txt file to be formatted
    # Path is relative to script
    fname = sys.argv[1]
    outfname = f"{fname.split('.')[0]}_formatted.txt"
    with open(fname, 'r') as f:
        latex_table = f.read()

    formatted = format_floats_in_latex_table(latex_table)
    with open(outfname, 'w') as f:
        f.write(formatted)
