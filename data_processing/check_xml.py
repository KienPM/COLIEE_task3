""" Create by Ken at 2020 Dec 24 """
file = open('/media/ken/Temp/TrainingData/COLIEE_Task3/COLIEE2020statute_data-English/train/riteval_H27_en.xml')
data = file.read()

close_tag_pos_set = set()
cur = 0
while True:
    cur = data.find('<pair', cur + 1)
    if cur == -1:
        break

    close_tag_pos = data.find('</pair', cur)
    if close_tag_pos in close_tag_pos_set:
        print(data[close_tag_pos:close_tag_pos + 50])

    close_tag_pos_set.add(close_tag_pos)
