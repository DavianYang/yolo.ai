from addict import Dict

class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)
    
    def __getattr__(self, name):
        try:
            val = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute {name}")
        except Exception as e:
            ex = e
        else:
            return val
        raise ex
            
            
class Config(object):
    def __init__(
        self,
        cfg_dict = None,
        cfg_text = None,
        filename = None
    ) -> None:
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')
        
        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)
        
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)
        