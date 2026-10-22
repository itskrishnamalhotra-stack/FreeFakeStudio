import fs from 'node:fs';
const path = new URL('./FreeFakeStudio.ipynb', import.meta.url);
const notebook = JSON.parse(fs.readFileSync(path, 'utf8'));
notebook.cells = [{cell_type:'code',execution_count:null,metadata:{cellView:'form',id:'freefakestudio-launch'},outputs:[],source:fs.readFileSync(new URL('./notebook_cell.py',import.meta.url),'utf8').split(/(?<=\n)/)}];
fs.writeFileSync(path, JSON.stringify(notebook,null,1)+'\n');
