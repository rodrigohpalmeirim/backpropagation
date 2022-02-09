from random import randint, uniform
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class Shape:
    def __init__(self, background_size, rel_size, rel_pos, outline, noise_amount):
        self.background_size = background_size
        self.image = Image.new("1", (background_size, background_size))
        self.draw = ImageDraw.Draw(self.image)
        self.outline = max(int(background_size * outline), 1)
        self.noise_amount = noise_amount
        self.x = (background_size - (rel_size[0] * background_size)) * rel_pos[0]
        self.y = (background_size - (rel_size[1] * background_size)) * rel_pos[1]
        self.width = rel_size[0] * background_size - 1
        self.height = rel_size[1] * background_size - 1
        self.draw_shape()
        self.draw_noise()
    
    def draw_shape(self):
        pass
    
    def draw_noise(self):
        for _ in range(int(self.background_size**2 * self.noise_amount)):
            x = randint(0, self.background_size - 1)
            y = randint(0, self.background_size - 1)
            self.image.putpixel((x, y), self.image.getpixel((x, y)) ^ 1)

class Ellipse(Shape):
    def draw_shape(self):
        self.draw.ellipse((self.x, self.y, self.x + self.width, self.y + self.height), fill=0, outline=self.outline, width=self.outline)

class Rectangle(Shape):
    def draw_shape(self):
        self.draw.rectangle((self.x, self.y, self.x + self.width, self.y + self.height), fill=0, outline=self.outline, width=self.outline)

class Triangle(Shape):
    def draw_shape(self):
        self.draw.line([
            (self.x + self.width / 2, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height),
            (self.x + self.width / 2, self.y)
        ], fill=1, width=self.outline)

class Cross(Shape):
    def draw_shape(self):
        self.draw.line((self.x + self.width / 2, self.y, self.x + self.width / 2, self.y + self.height), fill=1, width=self.outline)
        self.draw.line((self.x, self.y + self.height / 2, self.x + self.width, self.y + self.height / 2), fill=1, width=self.outline)


class Dataset:
    def __init__(self, image_count, background_size, shapes, size_variation, pos_variation, outline, noise_amount, training_ratio, validation_ratio, test_ratio, flatten=False):
        self.background_size = background_size
        self.image_count = image_count
        
        images = []
        labels = []
        for i in range(self.image_count):
            shape_size = (0.5 + uniform(-0.5, 0.5) * size_variation, 0.5 + uniform(-0.5, 0.5) * size_variation)
            shape_pos = (0.5 + uniform(-0.5, 0.5) * pos_variation, 0.5 + uniform(-0.5, 0.5) * pos_variation)
            shape = shapes[i % len(shapes)](background_size, shape_size, shape_pos, outline, noise_amount).image.getdata()
            if not flatten:
                shape = np.array(shape).reshape(background_size, background_size)
            images.append(np.array(shape))
            label = [0] * len(shapes)
            label[i % len(shapes)] = 1
            labels.append(label)
        
        training_count = int(self.image_count * training_ratio)
        validation_count = int(self.image_count * validation_ratio)
        test_count = int(self.image_count * test_ratio)

        self.training_set = np.array(images[:training_count])
        self.training_labels = np.array(labels[:training_count])
        self.validation_set = np.array(images[training_count:training_count + validation_count])
        self.validation_labels = np.array(labels[training_count:training_count + validation_count])
        self.test_set = np.array(images[training_count + validation_count:])
        self.test_labels = np.array(labels[training_count + validation_count:])
    
    def show_data(self, count=25):
        count = min(count, len(self.training_set))
        columns = int(np.ceil(np.sqrt(count)))
        lines = int(np.ceil(count / columns))
        f, axarr = plt.subplots(lines, columns)
        for i in range(columns):
            for j in range(lines):
                if i + j * columns < count:
                    axarr[j][i].imshow(self.training_set[i + j * columns].reshape(self.background_size, self.background_size), cmap='gray')
                axarr[j, i].axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    Dataset(500, 16, [Ellipse, Rectangle, Triangle, Cross], 0.5, 0.5, 0.05, 0.02, 0.70, 0.2, 0.1).show_data()