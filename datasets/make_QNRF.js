const fs = require('fs');

console.log('Beginning Image List Generation');

let stage = 'Test';
let in_file = `/drive/datasets/UCF-QNRF_ECCV18/list.txt`;
let out_file = `UCF-QNRF/${stage}.json`;

console.log('Reading Image Names');
let data = fs.readFileSync(in_file, {encoding: 'utf8', flag: 'r'});

data = data.split('\n');
let vals = [];
for (let d of data) { 
	let path = d.split(', ')[0];
	if (path.includes(stage))
		vals.push(path);
}

let out = [];
for (let v of vals) {
	out.push(`/drive/datasets/UCF-QNRF_ECCV18/${v}`);
}

out = JSON.stringify(out);
fs.writeFileSync(out_file, out);

console.log('Done Generating Image List');
