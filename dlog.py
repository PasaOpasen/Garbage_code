def _close_FileHandlers():
    for logger in _all_loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

def _open_FileHandlers():
    for logger in _all_loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler._open()



def process_filehandlers():
    """
    до вызова функции открывает файлы логов, а после вызова закрывает
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                _open_FileHandlers()
                return func(*args, **kwargs)
            finally:
                _close_FileHandlers()

        return wrapped

    return decorator











#region SESLOG

# этот кусок кода позволяет записывать
# каждую сессию вызова do_ocr с отладочной информацией
# в отдельный файл, куда идёт stdout и stderr
#
# его можно просто закомментировать, если такой функционал не требуется

# from dreamocr.utils.log.tee import Tee
# from dreamocr.utils.log.loggers import _set_StreamHandlers_to_stdout
#
# def _kwargs_to_tee_file(kw: dict):
#     global LOGS
#     tempdir = kw.get('tempdir')
#     if not tempdir:
#         return os.path.join(
#             LOGS,
#             'tee',
#             Path(kw['input_file']).stem + '_tee.txt',
#         )
#
#     return os.path.join(
#         tempdir,
#         'tee.txt'
#     )
#
# do_ocr = Tee.decorator(
#     file_from_kwargs=_kwargs_to_tee_file,
#     steams_corrector=_set_StreamHandlers_to_stdout,
#     file_mode='at'
# )(do_ocr)


#endregion

