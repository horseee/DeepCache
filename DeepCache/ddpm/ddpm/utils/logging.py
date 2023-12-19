import os
import sys
import time
import codecs
import logging


class Logger():
    def __init__(self, config, root_dir = 'runtime_log', 
                sub_name = None, overwrite = False, append=False):   
        self.log_dir = root_dir
        self.overwrite = overwrite

        self.format = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                        "%Y-%m-%d %H:%M:%S")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        if sub_name is None:
            sub_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.setup_sublogger(sub_name, config, append=append)

    def setup_sublogger(self, sub_name, sub_config, append=False):
        self.sub_dir = os.path.join(self.log_dir, sub_name)
   
        if append:
            os.makedirs(self.sub_dir, exist_ok=True)
        else:
            create_dir = True
            if os.path.exists(self.sub_dir):
                overwrite = False
                response = input(
                    f"Logging folder {self.sub_dir} already exists. Overwrite? (Y/N)"
                )
                if response.upper() == "Y":
                    overwrite = True
                else:
                    raise EnvironmentError("No overwriting stop the program. Consider change a path or overwrite the content (Better NOT)")

                if overwrite:
                    import shutil
                    shutil.rmtree(self.sub_dir)
            
            if create_dir:
                os.makedirs(self.sub_dir)

            self.write_description_to_folder(os.path.join(self.sub_dir, 'description.txt'), sub_config)
            with open(os.path.join(self.sub_dir, 'train.sh'), 'w') as f:
                f.write('python ' + ' '.join(sys.argv))

        # Setup File/Stream Writer
        log_format=logging.Formatter("%(asctime)s - %(levelname)s :       %(message)s", "%Y-%m-%d %H:%M:%S")
        
        self.writer = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.sub_dir, "training.log"))
        fileHandler.setFormatter(log_format)
        self.writer.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(log_format)
        self.writer.addHandler(consoleHandler)
        
        self.writer.setLevel(logging.INFO)

        # Checkpoint
        self.checkpoint_path = os.path.join(self.sub_dir, 'ckpt') 
        os.makedirs(self.checkpoint_path, exist_ok=True)   
        self.image_ckpt_path = os.path.join(self.sub_dir, 'img_ckpt')
        os.makedirs(self.image_ckpt_path, exist_ok=True)
        self.lastest_checkpoint_path = os.path.join(self.sub_dir, 'latest_model.bin')   

    def setup_image_folder(self, image_folder):
        image_dir = os.path.join(self.sub_dir, image_folder)
        if os.path.exists(image_dir):
            overwrite = True
            '''
            response = input(
                f"Image folder {image_dir} already exists. Overwrite? (Y/N)"
            )
            if response.upper() == "Y":
                overwrite = True
            '''
            
            if overwrite:
                import shutil
                shutil.rmtree(image_dir)
                os.makedirs(image_dir)
            
        else:
            os.makedirs(image_dir)
        
        with open(os.path.join(self.sub_dir, 'sample.sh'), 'w') as f:
            f.write('python ' + ' '.join(sys.argv))
        self.image_dir = image_dir

        return image_dir


    def log(self, info):
        self.writer.info(info)

    def write_description_to_folder(self, file_name, config):
        with codecs.open(file_name, 'w') as desc_f:
            desc_f.write("- Training Parameters: \n")
            for key, value in config.items():
                desc_f.write("  - {}: {}\n".format(key, value))

class EmptyLogger():
    def __init__(self, root_dir = 'runtime_log', 
                sub_name = None, overwrite = False, append=False):   
        self.log_dir = root_dir
        self.sub_dir = os.path.join(self.log_dir, sub_name)
        self.checkpoint_path = os.path.join(self.sub_dir, 'ckpt') 

    def log(self, info):
        pass

    def setup_image_folder(self, image_folder):
        image_dir = os.path.join(self.sub_dir, image_folder)
        self.image_dir = image_dir

        return image_dir