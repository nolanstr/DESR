from genepy.generator import Generator

generator = Generator()
generator.add_operator("add")
generator.add_operator("sub")
generator.add_operator("mult")
generator.add_operator("sin")
generator.add_operator("cos")

equation = generator()

import pdb;pdb.set_trace()
