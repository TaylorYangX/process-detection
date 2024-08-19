import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.exir import to_edge
from executorch.sdk import BundledProgram

from executorch.sdk.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.sdk.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torch.export import export


# Generate Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()

# Step 1: ExecuTorch Program Export
m_name = "forward"
method_graphs = {m_name: export(model, (torch.randn(1, 1, 32, 32),))}

# Step 2: Construct Method Test Suites
inputs = [[torch.randn(1, 1, 32, 32)] for _ in range(2)]

method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(inputs=inp, expected_outputs=getattr(model, m_name)(*inp))
            for inp in inputs
        ],
    )
]

# Step 3: Generate BundledProgram
executorch_program = to_edge(method_graphs).to_executorch()
bundled_program = BundledProgram(executorch_program, method_test_suites)

# Step 4: Serialize BundledProgram to flatbuffer.
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
    bundled_program
)
save_path = "bundled_program.bp"
with open(save_path, "wb") as f:
    f.write(serialized_bundled_program)