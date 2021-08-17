const fs = require('fs');

const PATH = "shanghai/part_B_train.json"

let data = fs.readFileSync(PATH, {flag: 'r'});
data = JSON.parse(data);

for (let i = 0; i < data.length; i++) {
	data[i] = data[i].replace('part_B', 'part_B_final');
}

fs.writeFileSync(PATH, JSON.stringify(data));
