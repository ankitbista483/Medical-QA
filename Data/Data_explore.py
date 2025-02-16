import xml.etree.ElementTree as ET
import os
import csv


dataset_directory = '/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/1_CancerGov_QA'
files = os.listdir(dataset_directory)

csv_file_path = '/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/data.csv'
xml_files = [file for file in files if file.endswith('.xml')]

with open(csv_file_path,mode='w',newline= '', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["Question","Answer"])

    for file in xml_files:
        file_path = os.path.join(dataset_directory,file)
       
        tree = ET.parse(file_path)
        root = tree.getroot()


        for _,qa in enumerate(root.findall(".//QAPair")):
            Question = qa.find('Question').text if qa.find('Question') is not None else "No Question"
            Answer = qa.find('Answer').text if qa.find('Answer') is not None else "No Answer"

            writer.writerow([Question,Answer])


print(f"Data has been saved in csv at {csv_file_path}")