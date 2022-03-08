
from typing import Sequence, Dict

import os, sys
import tempfile

# TODO remove

all_readers = ('miner', 'fitz', 'poppler')


import argparse

parser = argparse.ArgumentParser(
    prog='dreamocr.show_text',
    description='Make pdf text layer visible',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--in-pdf', required=True, type=str, help='Pdf file to process get layer', dest='inpath')
parser.add_argument('--out-path', required=True, type=str, help='Path to result files', dest='target_dir')
parser.add_argument('--reader', nargs='*', choices=list(all_readers),
                    default=[all_readers[0]],
                    help="Engine to read text layer"
                    )

parser.add_argument('--do-layout', action='store_true', help='Use our layout algorithm', dest='do_layout')
parser.add_argument('--show-order', action='store_true', help='Create files with order of tokens', dest='show_order')
parser.add_argument('--save-text', action='store_true', help='Save text layer to text file', dest='save_text')


def check_parsed_args(parsed):
    from dreamocr.utils.collectors.times import TimeCollector

    TC = TimeCollector(prefix='show_text-->', name='show_text')

    from dreamocr.utils.files import (is_pdf_file, is_normal_pdf, doc_name, mkdir)

    TC.done_task('import utils', False)

    assert is_pdf_file(parsed.inpath), f"input path {parsed.inpath} is not existing pdf file!"
    assert is_normal_pdf(parsed.inpath), f"input file is corrupted!"
    #assert os.path.isdir(parsed.target_dir), f"output path {parsed.target_dir} is not a directory!"
    mkdir(parsed.target_dir)

    # remove repeating engines with order saving
    parsed.reader = list(dict.fromkeys(parsed.reader))

    output_name = doc_name(parsed.inpath)
    parsed.temp_path = os.path.join(
        #os.path.dirname(parsed.outpath),
        tempfile.gettempdir(),
        output_name
    )
    mkdir(parsed.temp_path)

    return parsed, TC



def main(parsed, TC):

    from dreamocr.utils.pdf import gen_empty_pdf_gs, overlay_pdfs_fitz
    from dreamocr.types.json_for import Doc2
    from dreamocr.text_layer.utils import save_pdf_from_Doc2, plot_rectangles_pdf

    TC.done_task('import main modules')

    def get_json(file: str, engine: str, do_layout: bool):
        if do_layout:
            from dreamocr.text_layer.jsons.layout_json import to_json
            js = to_json(
                file,
                engine,
                parsed.temp_path,
                base_pdf_path=file,

                TC_global=TC
            )
        else:
            TC.init()

            if engine == 'miner':
                from dreamocr.text_layer.jsons.miner_json import to_json
                js = to_json(file)
            elif engine == 'fitz':
                from dreamocr.text_layer.jsons.fitz_json import to_json
                js = to_json(file, parsed.temp_path, file)

            else: # poppler
                from dreamocr.text_layer.layout.types.conversion.poppler_text import file_to_data_poppler
                js = file_to_data_poppler(file, 'v2').toDoc2()

            TC.done_task(f"{engine} read layer")


        if engine in ('fitz',):
            for tok in js.tokens:
                tok.text = tok.text.rstrip()

        return js

    jsons: Dict[str, Doc2] = {engine: get_json(parsed.inpath, engine, parsed.do_layout) for engine in parsed.reader}
    text_paths = {engine: os.path.join(parsed.temp_path, f"{engine}_text.pdf") for engine in parsed.reader}
    fore_text_paths = {engine: os.path.join(parsed.target_dir, f"{engine}_text.pdf") for engine in parsed.reader}

    TC.init()

    if parsed.save_text:
        for engine, doc in jsons.items():
            doc.save_as_text(os.path.join(parsed.target_dir, f"{engine}.txt"))
        TC.done_task('save as text')

    empty_path = os.path.join(parsed.temp_path, 'empty.pdf')
    gen_empty_pdf_gs(parsed.inpath, empty_path)
    TC.done_task('generate empty pdf')

    for engine, doc in jsons.items():
        save_pdf_from_Doc2(doc, text_paths[engine], alpha=0.8)
    TC.done_task(f"plot text of {parsed.reader}")

    for engine, doc in jsons.items():
        overlay_pdfs_fitz(empty_path, text_paths[engine], fore_text_paths[engine])

    TC.done_task(f"merge text and empty of {parsed.reader}")


    if parsed.show_order:

        order_paths = {engine: os.path.join(parsed.temp_path, f"{engine}_rectangles.pdf") for engine in parsed.reader}
        foreorder_paths = {engine: os.path.join(parsed.target_dir, f"{engine}_rectangles.pdf") for engine in parsed.reader}

        for engine, doc in jsons.items():
            plot_rectangles_pdf(doc, order_paths[engine], 0.4)
        TC.done_task(f"plot rects of {parsed.reader}")

        for engine, doc in jsons.items():
            overlay_pdfs_fitz(empty_path, order_paths[engine], foreorder_paths[engine])
        TC.done_task(f"merge rect and empty of {parsed.reader}")



    TC.show(3)

    pass



def full_main(args: Sequence[str]):
    parsed = parser.parse_args(args)
    parsed, TC = check_parsed_args(parsed)

    # вынесен, чтобы не делать лишних импортов до окончания проверок
    # и если вызывается с --help, не отнимало время
    main(parsed, TC)

    import shutil
    shutil.rmtree(parsed.temp_path)




def test_args(

        path_to_input: str = '../../data/files/docs/graficheskaia-razmetka/root/Арест_433.pdf',
        target_dir: str = 'debug_outputs/show_text/',
        reader: Sequence[str] = ('miner', 'fitz', 'poppler'),
        do_layout: bool = False,
        show_order: bool = True,
        save_text: bool = True
):

    # converts False to empty string
    check_flag = lambda prefix, arg: (prefix if arg else '')

    strings = (
        f"--in-pdf {path_to_input}",
        f"--out-path {target_dir}",
        f'--reader {" ".join(reader)}',
        check_flag('--do-layout', do_layout),
        check_flag('--show-order', show_order),
        check_flag('--save-text', save_text)
    )

    return ' '.join(strings).split()


def do_show_text(
        path_to_input: str = '../../data/files/docs/graficheskaia-razmetka/root/Арест_433.pdf',
        target_dir: str = 'debug_outputs/show_text/',
        reader: Sequence[str] = ('miner', 'fitz', 'poppler'),
        do_layout: bool = False,
        show_order: bool = True,
        save_text: bool = True
):
    args = test_args(
        path_to_input,
        target_dir,
        reader,
        do_layout,
        show_order,
        save_text
    )

    parsed = parser.parse_args(args)

    parsed, TC = check_parsed_args(parsed)

    main(parsed, TC)





if __name__ == '__main__':

    # из реального вызова
    sys.path.append(
        os.path.dirname(os.getcwd())
    )

    args = sys.argv[1:] if len(sys.argv) > 1 else test_args()


    full_main(args)


