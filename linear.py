import csv

def replace_semicolon_with_comma(input_file, output_file):
    with open(input_file, 'r', newline='') as csv_in:
        with open(output_file, 'w', newline='') as csv_out:
            reader = csv.reader(csv_in, delimiter=';')
            writer = csv.writer(csv_out, delimiter=',')
            for row in reader:
                writer.writerow(row)

input_file = './winequality/winequality-white.csv'  # 输入文件名
output_file = './winequality/winequality-white1.csv'  # 输出文件名

replace_semicolon_with_comma(input_file, output_file)
print("替换完成！")
