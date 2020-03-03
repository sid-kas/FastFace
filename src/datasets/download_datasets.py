import os, sys, json

parent_folder_path = os.path.dirname( os.path.abspath(__file__)).split(r'src')[0] # get parent folder
sys.path.append(parent_folder_path)

from src.tools.downloader import Downloader



class DatasetDownloader:
    def __init__(self, target_path="."):
        super().__init__()
        self.target_path = target_path
        self._ensure_dir(self.target_path)
        self.load_json()
        self.downloader = Downloader(target_path)
    
    def load_json(self):
        self.urls = []
        with open("src/datasets/datasets.json", 'r') as ds:
            datasets_info = json.load(ds)
        for k,v in datasets_info.items():   
            if v['relevant']:
                print(k)
                print(v['description'])
                self.urls.append(v['dataset_url'])
    
    def _ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download_and_extract(self):
        for url in self.urls:
            file_name = str(url).split('/')[-1]
            self.downloader.download_file_from_web_server(url,destination=self.target_path)
            self.downloader.extract_any_file(zip_file_path=os.path.join(self.target_path,file_name), destination=self.target_path)

dd = DatasetDownloader(target_path="src/datasets/")

dd.download_and_extract()