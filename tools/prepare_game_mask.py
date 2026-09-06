#!/usr/bin/env python3
"""Generate local physics image data; never download or bundle game assets."""
import argparse
import hashlib
from pathlib import Path

MASK_SHA256 = '6dac49cc6f6b02b3c9684a987dd32d507b9dda63f463665745d24113c4c01e70'
TGA_SHA256 = '5e7e018c624367789c543c663d85b8d0e1827a6b2be17664c288059cbd80cfd8'


def validate_mask(data):
    if len(data) != 16384 or hashlib.sha256(data).hexdigest() != MASK_SHA256:
        raise ValueError('mask does not match the supported Forever 2.11.26 image')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['extract', 'header'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = args.input.read_bytes()
    if args.mode == 'extract':
        if hashlib.sha256(data).hexdigest() != TGA_SHA256:
            parser.error('input is not the supported TestMaterialHeight.tga')
        # Pillow applies TGA origin/orientation and handles its compression.
        from PIL import Image
        with Image.open(args.input) as im:
            if im.size != (128, 128):
                parser.error('expected a 128x128 image')
            # The game's bitmap keeps bottom-up rows, unlike Pillow's display order.
            data = im.convert('RGB').transpose(Image.Transpose.FLIP_TOP_BOTTOM).getchannel('R').tobytes()
    try:
        validate_mask(data)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == 'extract':
        args.output.write_bytes(data)
    else:
        args.output.write_text('/* Generated locally. Do not redistribute. */\n'
            'static const uint8_t TMNF_FAKE_CONTACT_MASK[16384] = {\n' +
            ',\n'.join(','.join(str(x) for x in data[i:i+32])
                       for i in range(0, len(data), 32)) + '\n};\n')


if __name__ == '__main__':
    main()
