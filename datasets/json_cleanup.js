const fs = require('fs');

let file_path = 'UCF-QNRF/Train.json';

let data = fs.readFileSync(file_path);
data = JSON.parse(data);

for (let i = data.length-1; i >= 0; i--) {
	let d = data[i];
	if (!(/\d/.test(d))) {
		data.splice(i, 1);
	} else {
		data[i] = d.replace('/home/16amf8/data', '/drive');
	}
}

fs.writeFileSync(file_path, JSON.stringify(data));
