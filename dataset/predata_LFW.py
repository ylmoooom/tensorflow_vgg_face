import os
import shutil

traindir = "./train/"
if not os.path.exists(traindir):
	os.mkdir(traindir)

testdir = "./test"
if not os.path.exists(testdir):
	os.mkdir(testdir)

traintext = open("train.txt", "w")
testtext = open("test.txt", "w")

countpeople = 0
counttest = 0
counttrain = 0
for line in open("./lfw-names.txt"):
	count = line.split("\t")
	test_flag = 1
	if (int(count[1])) >= 10:
		src = os.path.join("lfw-aligned/", count[0])
		dst = os.path.join(traindir, count[0])
		dst2 = os.path.join(testdir, count[0])
		os.mkdir(dst)
		os.mkdir(dst2)
		for root, dirs, files in os.walk(src):
			for f in files:
				if (test_flag % 10 == 0):
					shutil.copy2(os.path.join(root,f), dst2)
					testtext.write(os.path.join(count[0], f) + "\t" + str(countpeople) + "\n")
					counttest += 1
				else:
					shutil.copy2(os.path.join(root,f), dst)
					traintext.write(os.path.join(count[0], f) + "\t" + str(countpeople) + "\n")
					counttrain += 1
				test_flag += 1
		countpeople += 1

traintext.close()
testtext.close()

print("train samples: " + str(counttrain) + "\n")
print("test samples: " + str(counttest) + "\n")
