import os
import subprocess

# See https://github.com/kynan/nbstripout

path = 'notebooks'

for dir, subdir, files in os.walk(path):
	for file in files:
		if file.endswith((".ipynb")):
			name = os.path.join(dir, file)
			subprocess.Popen('nbstripout ' + name, shell=True,
							   stdout=subprocess.PIPE,
							   stderr=subprocess.PIPE)