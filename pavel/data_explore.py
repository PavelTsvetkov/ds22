import pandas as pd
import pickle as pic

from pavel.utils import pre_process, detectClasses, count_words_in_column, TrainingGenerator

print("Loading dataset")
dataset = pd.read_csv("C:\\tmp\\dabble\\movies_metadata.csv")

print("Preprocessing")
dataset = pre_process(dataset)  # lower case, cleanse, etc.

print("Detecting classes")
# dataset, class_count = detectClasses(dataset, column="genres", prefix="gen_")  # generates new columns, one per class

print("Counting words")
word_stats = count_words_in_column(dataset["overview"])

# with open("C:\\tmp\\dabble\\test.dict", mode="wb") as f:
#             pic.dump(word_stats, f, pic.HIGHEST_PROTOCOL)
with open("C:\\tmp\\dabble\\words.txt", mode="w",encoding="utf-8") as f:
    f.writelines([key+","+str(value)+"\r\n" for key,value in word_stats.items()])



# with(open("C:\\tmp\\dabble\\custom_synonyms.txt")) as sf:
#     synonyms = {line.strip().split(",")[0]: line.strip().split(",")[1] for line in sf}
#
# generator = TrainingGenerator(maxLen=140, model_file="C:\\tmp\\dabble\\GoogleNews-kvectors.bin", synonim_file="C:\\tmp\\dabble\\custom_synonyms.txt")
#
# with open("C:\\tmp\\dabble\\test.dict", mode="rb") as f:
#     word_stats = pic.load(f)
#
# sorted_d = sorted(
#     ((value, key) for (key, value) in word_stats.items() if not generator.can_convert(key)),
#     reverse=True)
#
# print(sorted_d)
