import pygame as pg
import moderngl as mgl


class Textures:
    def __init__(self, app):
        self.app = app
        self.ctx = self.app.ctx

        # load texture
        self.texture_0 = self.load('test.png')

        # assign texture unit
        self.texture_0.use(location=0)

    def load(self, file_name):
        texture = pg.image.load(f'assets/{file_name}')
        texture = pg.transform.flip(texture, flip_x=True, flip_y=False)
        texture = self.ctx.texture(texture.get_size(), 4,
                                   pg.image.tobytes(texture, 'RGBA'))
        texture.anisotropy = self.ctx.max_anisotropy
        texture.build_mipmaps()
        # texture.filter == (mgl.NEAREST, mgl.NEAREST)
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        return texture






