

import os
from pathlib import Path

import shutil

from PIL import Image

from pdf2image import convert_from_path




for class_ in ('11А', '11Б', '11В', '11Г'):

    class_result_path = Path(os.path.join('images', class_))

    for file in os.listdir(os.path.join(Path(os.getcwd()).parents[0], class_)):

        if file.endswith('.pdf'):

            pages = convert_from_path(
                os.path.join(class_, file),
                grayscale = True,
                dpi = 150
            )

            for i, p in enumerate(pages):

                p.save(
                    os.path.join(class_result_path, Path(file).stem + f"_{i+1}.png")
                )
    else:

        shutil.copyfile(
            os.path.join(class_, file),
            os.path.join(class_result_path, file)
        )










