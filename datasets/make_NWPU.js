const fs = require('fs');

console.log('Beginning Image List Generation');

let stage = 'val';
let in_file = `/drive/datasets/NWPU-Crowd/${stage}.txt`;
let out_file = `NWPU/${stage}.json`;

console.log('Reading Image Names');
let data = fs.readFileSync(in_file, {encoding: 'utf8', flag: 'r'});

data = data.split('\n');
let vals = [];
for (let d of data) 
	vals.push(d.split(' ')[0]);

let out = [];
for (let v of vals) {
	out.push(`/drive/datasets/NWPU-Crowd/images/${v}.jpg`);
}

out = JSON.stringify(out);
fs.writeFileSync(out_file, out);

console.log('Done Generating Image List');
