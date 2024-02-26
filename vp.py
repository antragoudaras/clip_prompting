"""Defines various kinds of visual-prompting modules for images."""
import torch
import torch.nn as nn
import numpy as np


class FixedPatchPrompter(nn.Module):
    """
    Defines visual-prompt as a fixed patch over an image.
    For refernece, this prompt should look like Fig 2(a) in the PDF.
    """
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()

        assert isinstance(args.image_size, int), "image_size must be an integer"
        assert isinstance(args.prompt_size, int), "prompt_size must be an integer"

       
        # TODO: Define the prompt parameters here. The prompt is basically a
        # patch (can define as self.patch) of size [prompt_size, prompt_size]
        # that is placed at the top-left corner of the image.

        # Hints:
        # - The size of patch needs to be [1, 3, prompt_size, prompt_size]
        #     (1 for the batch dimension)
        #     (3 for the RGB channels)
        # - You can define variable parameters using torch.nn.Parameter
        # - You can initialize the patch randomly in N(0, 1) using torch.randn
        self.device, self.prompt_size = args.device, args.prompt_size
        self.patch = nn.Parameter(torch.randn(1, 3, self.prompt_size, self.prompt_size), requires_grad=True)

    def forward(self, x):
        # TODO: For a given batch of images, place the patch at the top-left

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        prompt = torch.zeros_like(x)
        prompt[:, :, :self.prompt_size, :self.prompt_size] = self.patch
        prompt = prompt.to(self.device)
        x = x.to(self.device)
        return x + prompt

class PadPrompter(nn.Module):
    """
    Defines visual-prompt as a parametric padding over an image.
    For refernece, this prompt should look like Fig 2(c) in the PDF.
    """
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Define the padding as variables self.pad_left, self.pad_right, self.pad_up, self.pad_down

        # Hints:
        # - Each of these are parameters that we need to learn. So how would you define them in torch?
        # - See Fig 2(c) in the assignment to get a sense of how each of these should look like.
        # - Shape of self.pad_up and self.pad_down should be (1, 3, pad_size, image_size)
        # - See Fig 2.(g)/(h) and think about the shape of self.pad_left and self.pad_right
        self.device = args.device
        self.pad_up = nn.Parameter(torch.randn(1, 3, pad_size, image_size), requires_grad=True)
        self.pad_down = nn.Parameter(torch.randn(1, 3, pad_size, image_size), requires_grad=True)
        self.pad_right = nn.Parameter(torch.randn(1, 3, image_size - 2*pad_size, pad_size), requires_grad=True)
        self.pad_left = nn.Parameter(torch.randn(1, 3, image_size - 2*pad_size, pad_size), requires_grad=True)
        self.pad_size = pad_size

    def forward(self, x):
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: For a given batch of images, add the prompt as a padding to the image.

        # Hints:
        # - First define the prompt. Then add it to the batch of images.
        # - It is always advisable to implement and then visualize if
        #   your prompter does what you expect it to do.

        prompt = torch.zeros_like(x)
        prompt[:,:, :self.pad_size, :] = self.pad_up
        prompt[:,:, x.shape[2] - self.pad_size:, :] = self.pad_down
        prompt[:, :, self.pad_size: x.shape[2] - self.pad_size, x.shape[3] - self.pad_size:] = self.pad_right
        prompt[:, :, self.pad_size: x.shape[2] - self.pad_size, :self.pad_size] = self.pad_left
        prompt = prompt.to(self.device)
        x = x.to(self.device)
        return x + prompt
        