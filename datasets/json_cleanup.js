const fs = require('fs');

let file_path = 'NWPU/val.json';

let data = fs.readFileSync(file_path);
data = JSON.parse(data);

for (let i = data.length-1; i >= 0; i--) {
	let d = data[i];
	if (!(/\d/.test(d))) {
		data.splice(i, 1);
	}
}

fs.writeFileSync(file_path, JSON.stringify(data));
