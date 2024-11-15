import logging
from logging.handlers import RotatingFileHandler
import os
import sys


def load_logger(args,
                logsave='out.log',
                do_log_print_to_screen=False,
                do_log_save_to_file=True):

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logname = os.path.join(args.output_dir, logsave)
    # StreamHandler
    if do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            datefmt='%Y/%m/%d %H:%M:%S', fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if do_log_save_to_file:
        file_handler = RotatingFileHandler(logname, maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(args):
            if not key.startswith("_"):
                config_dict[key] = getattr(args, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger
