
from typing import Optional, Callable, Dict, Any

import functools
import os
import pprint
import sys

from traceback import print_exc

from dreamocr.utils.files import mkdir_of_file

class Tee:
    """
    если написать
        sys.stdout = Tee(sys.stdout, file1, file2, ...)
    то print будет записывать не только в стандартный поток, но и в указанные файлы
    """
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()


    @staticmethod
    def decorator(
        filepath: Optional[str] = None,
        file_from_kwargs: Optional[Callable[[Dict[str, Any]], str]] = None,
        steams_corrector: Optional[Callable[[None],None]] = None,
        file_mode: str = 'w'
    ):
        """
        возвращает декоратор, который обертывает функцию так,
        чтобы стандартный вывод на время её работы уходил не только на консоль,
        но и в текстовый файл, причём есть возможность в зависимости от аргументов самой функции
        создавать свой файл

        @param filepath: фиксированный путь к файлу, куда будет записываться вывод
            (если задаём один раз)
        @param file_from_kwargs: функция, которая создаст путь к файлу, используя словарь kwargs
            (тогда можно сделать свой файл под разные комбинации аргументов)
        @param steams_corrector: функция без аргументов и без возврата, которая обновляет всякие другие потоки под новый sys.stdout
            (например, при изменении sys.stdout нужно обновить и логгеры)
        @param file_mode:
        """

        if filepath is None and file_from_kwargs is None:
            raise Exception("all file path args are None, no file to save output")

        fix_streams = steams_corrector if steams_corrector is not None else (lambda : None)
        kwargs_to_file = (lambda kw: filepath) if filepath is not None else file_from_kwargs

        def dec(func: Callable):

            @functools.wraps(func)
            def wrapper(*args, **kwargs):

                file_path = kwargs_to_file(kwargs)
                mkdir_of_file(file_path)
                file = open(file_path, mode = file_mode)

                saved_stdout = sys.stdout
                saved_stderr = sys.stderr
                sys.stdout = Tee(sys.stdout, file)
                sys.stderr = Tee(sys.stderr, file)

                try:
                    fix_streams()

                    print(f"pid: {os.getpid()}, cwd: {os.getcwd()}")
                    dct = {
                        'args': args,
                        'kwargs': kwargs
                    }
                    # если использовать чисто pprint.pprint,
                    # то вывод в поток будет кусочками,
                    # что в логах со многими процессами плохо выглядит
                    print(
                        '\n' + pprint.pformat(dct) + '\n\n'
                    )

                    return func(*args, **kwargs)

                except Exception as ex:

                    # как оказалось, блок finally запускается до выброса исключения,
                    # из-за чего само исключение не писалось в файл,
                    print_exc(file=file)
                    raise ex

                finally:
                    sys.stdout = saved_stdout
                    sys.stderr = saved_stderr
                    fix_streams()
                    file.close()



            return wrapper

        return dec