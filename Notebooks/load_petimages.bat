rem download kagglecatsanddogs_3367a.zip from https://www.microsoft.com/en-us/download/details.aspx?id=54765
python tools\load_image_data.py kagglecatsanddogs_3367a.zip cat:petimages\cat dog:petimages\dog --shape=50,50,3 --out=petimages %*
