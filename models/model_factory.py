def get_model(model_names):
    models = []
    for name in model_names:
        if name == "all":
            from models import cnn
            from models import Linear
            from models import net1
            return [cnn.CNN(), linear.Linear(), net1.Net1()]
        
        elif name.lower() == "cnn":
            from models import cnn
            models.append(cnn.CNN())
            
        elif name.lower() == "linear":
            from models import linear
            models.append(linear.Linear())
            
        elif name.lower() == "net1":
            from models import net1
            models.append(net1.Net1())
        
        else:
            raise NameError("Model name not found")
    return models