#!/usr/bin/env python3
"""Flatten manuscript and supplement into standalone submission TeX files."""
from pathlib import Path
import re, shutil
ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'paper'
main=PAPER/'main.tex'

def expand(path,seen=None):
    seen=seen or set()
    text=path.read_text(encoding='utf-8')
    base=path.parent
    conditional=re.compile(r'\\IfFileExists\{([^}]+)\}\{\\input\{([^}]+)\}\}\{\}')
    def resolve_conditional(m):
        name=m.group(2)
        for anchor in (PAPER, base, ROOT):
            candidate=anchor/name
            candidate=candidate if candidate.suffix else candidate.with_suffix('.tex')
            if candidate.exists():
                return f'\\input{{{name}}}'
        return ''
    text=conditional.sub(resolve_conditional,text)
    pat=re.compile(r'\\input\{([^}]+)\}')
    def repl(m):
        name=m.group(1)
        # TeX resolves input paths from the compilation working directory (paper/).
        candidates=[]
        for anchor in (PAPER, base, ROOT):
            p=anchor/name
            p=p if p.suffix else p.with_suffix('.tex')
            candidates.append(p)
        p=next((q for q in candidates if q.exists()), None)
        if p is None:
            return m.group(0)
        p=p.resolve()
        if p in seen:
            return '% recursive input omitted'
        return expand(p,seen|{p})
    return pat.sub(repl,text)

sub=ROOT/'submission'
sub.mkdir(parents=True,exist_ok=True)
out=sub/'manuscript_submission.tex'
out.write_text(expand(main,{main.resolve()}),encoding='utf-8')
supplement=PAPER/'supplement.tex'
supplement_out=sub/'Supplementary_Information.tex'
supplement_out.write_text(expand(supplement,{supplement.resolve()}),encoding='utf-8')
fig_src=ROOT/'paper/figures'; fig_dst=sub/'figures'
if fig_dst.exists(): shutil.rmtree(fig_dst)
fig_dst.mkdir(parents=True)
graphics=set()
for source in (out,supplement_out):
    content=source.read_text(encoding='utf-8')
    graphics.update(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',content))
for name in sorted(graphics):
    candidates=[fig_src/name,fig_src/'raw'/name]
    source=next((candidate for candidate in candidates if candidate.is_file()),None)
    if source is None:
        raise FileNotFoundError(f'Figure referenced by flattened TeX was not found: {name}')
    relative=source.relative_to(fig_src)
    destination=fig_dst/relative
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
print(out)
print(supplement_out)
