filename = "test1000_0.csv"

# csvファイルの読み込み
with open(filename, "r") as f:
    data = f.readline()
    data = f.readline()
    print(len(data.split(",")))
