import os

class Installer(object):
    def __init__(self):
        self.file_name = 'requirements.txt'
        self.prefix = 'pip install '
        self.uninstall_prefix = 'pip uninstall '
        # self.install()

    def install(self):
        with open(self.file_name, 'r') as f:
            content = f.readlines()
            for i, line in enumerate(content):
                # this line is only for the requirements that in the file now but will be excluded when goes to production
                if not line.startswith('--') and not line.startswith('#'):
                    if "PyIxExplorer" in line:
                        print(self.uninstall_prefix + "PyIxExplorer")
                        os.system(self.uninstall_prefix + "--yes PyIxExplorer")
                    print(self.prefix + line)
                    os.system(self.prefix+line)
                    # verify what we send to the cmd


if __name__ == "__main__":
    inst = Installer()
    inst.install()