import torch
import time
import torch.nn as nn
from util.mesh import simplify_mesh, compute_cotan_laplacian, compute_robust_laplacian

INPUT_DIM = 3
RGB_COLOR_DIM = 3
DISNEY_OUTPUT_DIM = 12

class EncodingLayer(nn.Module):
    """ Custom layer for multiresolution feature encoding and barycentric interpolation """
    def __init__(
            self,
            len_feature_vec,
            mesh,
            resolutions,
            reg_type='L1',
            use_zero_init=True,
        ):

        super().__init__()
        self.resolutions = resolutions
        self.num_resolutions = len(resolutions)
        self.len_feature_vec = len_feature_vec

        assert reg_type in ['L1', 'L2'], 'Invalid regularization type'
        self.reg_type = reg_type

        self.encoding_dim = len_feature_vec
        len_feature_vec = [len_feature_vec] * self.num_resolutions
        print(f"Mesh has {mesh.vertices.shape[0]} vertices")

        latent_features = []
        index_mappings = []
        for i, resolution in enumerate(resolutions):
            simple_mesh, index_mapping = simplify_mesh(mesh, resolution)
            if i == self.num_resolutions-1:  
                # Compute Laplacian
                L = compute_robust_laplacian(mesh.vertices, mesh.faces)
                self.register_buffer(f"laplacian", L)

            num_vertices = simple_mesh.vertices.shape[0]
            if not use_zero_init or i==0:
                a = 5e-4
                latent_features_mat = torch.normal(mean=0., std=a, size=(num_vertices, len_feature_vec[i]))
            else:
                latent_features_mat = torch.zeros(num_vertices, len_feature_vec[i])
            latent_features.append(nn.Parameter(latent_features_mat))
            index_mapping = torch.tensor(index_mapping, requires_grad=False)
            index_mappings.append(index_mapping)


        for i,m in enumerate(index_mappings):
            self.register_buffer(f"mapping_{i}",m)
            
        self.latent_features = nn.ParameterList(latent_features)

    @property
    def mappings(self):
        return [getattr(self, f"mapping_{i}") for i in range(self.num_resolutions)]

    def get_regularization(self, lambda_reg):
        laplacian = self.laplacian
        lat_feats = self.get_all_cat_features_finest_res()

        if self.reg_type == 'L2':
            regularization_term = lambda_reg * torch.sum(torch.sparse.mm(laplacian, lat_feats) ** 2)
        else:
            regularization_term = lambda_reg * torch.sum(torch.abs(torch.sparse.mm(laplacian, lat_feats)))

        return regularization_term
    

    def get_all_cat_features_finest_res(self):
        index_mappings = [getattr(self, f"mapping_{i}") for i in range(self.num_resolutions)]

        # Gather corner features in a more efficient way
        corner_features = [
            self.latent_features[i][index_mappings[i]] for i in range(self.num_resolutions)
        ]
        feature_vector = torch.sum(torch.stack(corner_features), dim=0)

        return feature_vector



    def forward(self, bary, triangle):
        assert bary.shape[0] == triangle.shape[0]  # batch size
        assert bary.shape[1] == INPUT_DIM # ideally 3
        assert triangle.shape[1] == 3

        index_mappings = [getattr(self, f"mapping_{i}") for i in range(self.num_resolutions)]

        corner_features = [
            self.latent_features[i][index_mappings[i][triangle]] for i in range(self.num_resolutions)
        ]
        feature_vector = torch.cat(corner_features, dim=2)
        feature_vector = torch.sum(torch.stack(corner_features), dim=0)

        assert bary.shape[0] == feature_vector.shape[0]
        assert feature_vector.shape[1] == INPUT_DIM

        feature_vecs = torch.matmul(bary.unsqueeze(1), feature_vector).squeeze(1)

        return feature_vecs



class MeshFeatModel(nn.Module):
    def __init__(self, 
        num_layers,
        mesh,
        resolutions,
        hidden_dim=64,  
        len_feature_vec=4,
        output_dim=RGB_COLOR_DIM,
        activation_func = nn.ReLU,
        reg_type='L2',
        use_zero_init=False,
        neural=True,
        **kwargs
        ):
        super(MeshFeatModel, self).__init__()

        # encoding of the features
        self.encoding_layer = EncodingLayer(
            len_feature_vec,
            mesh,
            resolutions,
            reg_type=reg_type,
            use_zero_init=use_zero_init,
        )

        # Define the layers of the MLP
        layers = []
        # input layer
        if neural:
            mlp_input_dim = self.encoding_layer.encoding_dim
            layers.append(
                nn.Sequential(
                    nn.Linear(mlp_input_dim, hidden_dim),
                    activation_func()
                )
            )
            # hidden layers
            for _ in range(num_layers):
                layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        activation_func()
                    )
                )
            # output layer
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    )
            )
        self.layers = nn.ModuleList(layers)
    
    def forward(self, bary, triangle):
        # encode the point using MeshFeat 
        features = self.encoding_layer(bary, triangle) 

        # forward pass through all the mlp layers
        res = features
        for i in range(len(self.layers)):
            res = self.layers[i](res)

        return torch.sigmoid(res)