import torch

class ShadingCoordinateSystem():
    @staticmethod
    def cos_theta(vectors):
        # Vectors need to be normalized for this to work. Do it for safety
        vectors = torch.nn.functional.normalize(vectors)
        return vectors[..., 2]
    
    @staticmethod
    def get_trafo_to_shading_cosy(normals):
        """ Constructions the transformation that transforms light and viewing direction into the shading coordinate system
        (https://www.pbr-book.org/3ed-2018/Reflection_Models#x0-GeometricSetting)
        The function constructs two tangential vectors b and t (i.e. both orthogonal to the normal) and uses
        t, b, and the normal to construct a rotation matrix.

        Remark: t is constructed as the cross product of the normal with [0, 0, 1]. This can be written explicitely
        therefore the computation is compressed. The choice [0, 0, 1] is arbitrary, any vector could be used
        (We want to obtain a vector that is orthogonal to the normal). We could also do
        t = light_dirs - normals * light_dirs
        """
        # Compute first tangent as cross product of the normal with (arbitrary) [0, 0, 1]
        t = torch.stack([normals[:, 1], -normals[:, 0], torch.zeros_like(normals[:, 0])], dim=1)
        t = torch.nn.functional.normalize(t, dim=-1)

        # Compute second tangent as cross product of normal with first tangent
        b = torch.cross(normals, t, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)

        # Construct rotation matrix by using t, b, and normal as rows of the matrix
        rot = torch.stack([t, b, normals], dim=1)

        return rot

    @staticmethod
    def transform_to_shading_cosy(normals, light_dirs, view_dirs):
        """ Function that transforms light and viewing direction into the shading coordinate system
        (https://www.pbr-book.org/3ed-2018/Reflection_Models#x0-GeometricSetting)

        Function expects batched input, i.e. input vectors must be [N x 3] (N needs to be the same for all)
        """

        # Get the transformations matrix to the shading coordinate systems
        trafos = ShadingCoordinateSystem.get_trafo_to_shading_cosy(normals)

        # Transform the vectors
        ldir = torch.bmm(trafos, light_dirs.unsqueeze(-1)).squeeze()
        vdir = torch.bmm(trafos, view_dirs.unsqueeze(-1)).squeeze()

        return ldir, vdir

    @staticmethod
    def get_trafo_to_shading_cosy_fun_handle(normals):
        """ Function that transforms light and viewing direction into the shading coordinate system
        (https://www.pbr-book.org/3ed-2018/Reflection_Models#x0-GeometricSetting)

        Function expects batched input, i.e. input vectors must be [N x 3] (N needs to be the same for all)
        """

        # Get the transformations matrix to the shading coordinate systems
        trafos = ShadingCoordinateSystem.get_trafo_to_shading_cosy(normals)

        # Define function for transformation
        def trafos_fun(vecs):
            return torch.bmm(trafos, vecs.unsqueeze(-1)).squeeze()

        return trafos_fun