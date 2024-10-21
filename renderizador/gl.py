#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Luciano Felix
Disciplina: Computação Gráfica
Data: 14/08/2014
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    transform_stack = []
    transform_matrix = np.identity(4)

    @staticmethod
    def setup(width, height, near=0.01, far=1000, samplerate=1):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far
        GL.samplerate = samplerate
        GL.transform_stack = []

        GL.screen_matrix = np.array([
            [width/2,       0.0, 0.0,  width/2],
            [    0.0, -height/2, 0.0, height/2],
            [    0.0,       0.0, 1.0,      0.0],
            [    0.0,       0.0, 0.0,      1.0],
        ])

        GL.width *= samplerate
        GL.height *= samplerate

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).
        coords = np.array(point).reshape(-1, 2).clip((0, 0), (GL.width - 1, GL.height - 1))
        color = (np.array(colors["emissiveColor"]) * 255).astype(np.int32)

        for coord in np.rint(coords).astype(np.uint32).tolist():
            gpu.GPU.draw_pixel(coord, gpu.GPU.RGB8, color)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        points = []
        anchors = np.array(lineSegments).reshape(-1, 2)

        for p0, p1 in zip(anchors[:-1], anchors[1:]):
            # Bresenham's line algorithm
            dp = p1 - p0
            dx, dy = np.abs(dp)
            sx, sy = np.sign(dp) / GL.samplerate
            x, y = p0

            if dx > dy:
                err = dx / 2.0

                while not np.isclose(x, p1[0], 0.05, 0.05):
                    points.extend((x * GL.samplerate, y * GL.samplerate))

                    err -= dy

                    if err < 0:
                        y += sy
                        err += dx

                    x += sx
            else:
                err = dy / 2.0

                while not np.isclose(y, p1[1], 0.05, 0.05):
                    points.extend((x * GL.samplerate, y * GL.samplerate))

                    err -= dx

                    if err < 0:
                        x += sx
                        err += dy

                    y += sy

        GL.polypoint2D(points, colors)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).        
        points = []
        bx = np.arange(0, radius * GL.samplerate)
        by = np.arange(0, radius * GL.samplerate)

        for i, j in np.ndindex(len(bx), len(by)):
            d = np.hypot(bx[i], by[j])
            r = radius * GL.samplerate
            if d <= r and d > r - GL.samplerate:
                points.extend((bx[i], by[j]))

        GL.polypoint2D(points, colors)

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        p = np.array(vertices).reshape(-1, 2) * GL.samplerate
        x, y = p.T
        vec = np.diff(p, axis=0, append=p[0:1])
        norm = vec @ np.array([[0, 1], [-1, 0]])
        bx = np.arange(np.clip(np.min(x), 0, GL.width - 1), np.clip(np.max(x), 0, GL.width - 1) + 1)
        by = np.arange(np.clip(np.min(y), 0, GL.height - 1), np.clip(np.max(y), 0, GL.height - 1) + 1)
        bx_grid, by_grid = np.meshgrid(bx, by, indexing="ij")
        coords = np.column_stack([bx_grid.ravel(), by_grid.ravel()])
        diffs = coords[:, np.newaxis, :] - p[np.newaxis, :, :]
        dot_products = np.einsum('ijk,jk->ij', diffs, norm)
        mask = np.all(dot_products <= 0, axis=1)

        GL.polypoint2D(coords[mask].ravel(), colors)

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        triangles = np.array(point).reshape(-1, 3, 3)

        for triangle in triangles:
            points = []

            for vertex in triangle:
                v = np.pad(vertex, (0, 1), mode="constant", constant_values=1.0)
                v = GL.viewpoint_matrix @ GL.transform_matrix @ v
                x, y, z, w = GL.screen_matrix @ (v / v[3])

                points.extend([x, y])

            GL.triangleSet2D(points[::1], colors)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        rotation = GL.transform_rotate(orientation)
        translation = GL.transform_translate([-n for n in position])
        look_at = np.linalg.inv(np.linalg.inv(translation) @ np.linalg.inv(rotation))

        fov_y = 2 * np.arctan(np.tan(fieldOfView / 2) * (GL.height / np.hypot(GL.width, GL.height)))
        top = GL.near * np.tan(fov_y)
        right = top * GL.width / GL.height

        x = GL.near / right
        y = GL.near / top
        z = ((-GL.near - GL.far) / (GL.near - GL.far))
        l = -2 * GL.far * GL.near / (GL.far - GL.near)

        perspective = np.array([
            [x, 0,  0, 0],
            [0, y,  0, 0],
            [0, 0,  z, l],
            [0, 0, -1, 0],
        ])

        GL.viewpoint_matrix = perspective @ look_at

    @staticmethod
    def transform_translate(translation):
        matrix = np.array([
            [1.0, 0.0, 0.0, translation[0]],
            [0.0, 1.0, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0,            1.0],
        ])

        return matrix

    @staticmethod
    def transform_scale(scale):
        matrix = np.array([
            [scale[0],      0.0,      0.0, 0.0],
            [     0.0, scale[1],      0.0, 0.0],
            [     0.0,      0.0, scale[2], 0.0],
            [     0.0,      0.0,      0.0, 1.0],
        ])

        return matrix

    @staticmethod
    def transform_rotate(orientation):
        theta = orientation[3]

        quaternion = np.array([
            np.cos(theta / 2),
            np.sin(theta / 2) * orientation[0],
            np.sin(theta / 2) * orientation[1],
            np.sin(theta / 2) * orientation[2],
        ])

        norm = np.linalg.norm(quaternion)
        r, i, j, k = quaternion / norm

        matrix = np.array([
            [1-2*(j*j+k*k),   2*(i*j-k*r),   2*(i*k+j*r), 0.0],
            [  2*(i*j+k*r), 1-2*(i*i+k*k),   2*(j*k-i*r), 0.0],
            [  2*(i*k-j*r),   2*(j*k+i*r), 1-2*(i*i+j*j), 0.0],
            [          0.0,           0.0,           0.0, 1.0],
        ])

        return matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.
        matrix = GL.transform_translate(translation) @ GL.transform_scale(scale) @ GL.transform_rotate(rotation)

        if len(GL.transform_stack) > 0:
            matrix = GL.transform_stack[-1] @ matrix

        GL.transform_matrix = matrix
        GL.transform_stack.append(GL.transform_matrix)

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if len(GL.transform_stack) == 0:
            return

        GL.transform_stack.pop()

        if len(GL.transform_stack) == 0:
            GL.transform_matrix = np.identity(4)
            return

        GL.transform_matrix = GL.transform_stack[-1]

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        offset = 0
        triangles = []

        for strip_size in stripCount:
            for index in range(strip_size - 2):
                i1 = (offset + index + 0) * 3
                i2 = (offset + index + 1) * 3
                i3 = (offset + index + 2) * 3
                triangle = [point[i1:i1+3], point[i2:i2+3], point[i3:i3+3]]

                if index % 2 != 0:
                    triangle = triangle[::-1]

                triangles.append(triangle)

            offset += strip_size

        GL.triangleSet(triangles, colors)

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        start = 0
        while start < len(index):
            try:
                end = index.index(-1, start)
            except ValueError:
                break

            indexes = index[start:end]
            i0 = indexes[0] * 3
            p0 = point[i0 : i0+3]

            triangles = []
            for i in range(1, len(indexes) - 1):
                i1 = indexes[i+0] * 3
                i2 = indexes[i+1] * 3
                p1 = point[i1 : i1+3]
                p2 = point[i2 : i2+3]

                triangles.append([p0, p1, p2])
            
            GL.triangleSet(triangles, colors)

            start = end + 1
        
        return

    @staticmethod
    def triangle_area(p0, p1, p2):
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2

        return np.abs(x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) / 2

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        print("coord", coord)
        print("coordIndex", coordIndex)
        print("colorPerVertex", colorPerVertex)
        print("color", color)
        print("colorIndex", colorIndex)
        print("texCoord", texCoord)
        print("texCoordIndex", texCoordIndex)
        print("colors", colors)
        print("current_texture", current_texture)

        if not colorPerVertex or not color or len(colorIndex) < 4:
            return GL.indexedTriangleStripSet(coord, coordIndex, colors)

        faces_coords = []
        faces_colors = []

        start = 0
        while start < len(coordIndex):
            try:
                end = coordIndex.index(-1, start)
            except ValueError:
                break

            indexes = coordIndex[start:end]
            i0 = indexes[0] * 3
            p0 = coord[i0 : i0+3]

            triangles = []
            for index in range(1, len(indexes) - 1):
                i1 = indexes[index] * 3
                i2 = indexes[index+1] * 3
                p1 = coord[i1 : i1+3]
                p2 = coord[i2 : i2+3]

                triangles.append([p0, p1, p2])

            faces_coords.append(triangles)

            base_color = np.array([color[i*3:(i+1)*3] for i in colorIndex[start:end]]).reshape(3, 3)

            faces_colors.append(np.roll(base_color, 1, axis=0))

            start = end + 1

        for (face, base_color) in zip(faces_coords, faces_colors):
            triangles = np.array(face).reshape(-1, 3, 3)

            for triangle in triangles:
                p = []

                for vertex in triangle:
                    v = np.pad(vertex, (0, 1), mode="constant", constant_values=1.0)
                    v = GL.viewpoint_matrix @ GL.transform_matrix @ v
                    p_vertex = GL.screen_matrix @ (v / v[3])

                    p.append(p_vertex[0:3])

                p = np.array(p) * GL.samplerate
                x, y, z = p.T
                vec = np.diff(p, axis=0, append=p[0:1])
                norm = vec @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                bx = np.arange(np.clip(np.min(x), 0, GL.width - 1), np.clip(np.max(x), 0, GL.width - 1) + 1)
                by = np.arange(np.clip(np.min(y), 0, GL.height - 1), np.clip(np.max(y), 0, GL.height - 1) + 1)
                bx_grid, by_grid = np.meshgrid(bx, by, indexing="ij")
                coords = np.column_stack([bx_grid.ravel(), by_grid.ravel()])
                diffs = coords[:, np.newaxis, :] - p[np.newaxis, :, :2]
                dot_products = np.einsum('ijk,jk->ij', diffs, norm[:, :2])
                mask = np.all(dot_products <= 0, axis=1)
                points = coords[mask]

                area = np.abs(np.dot(x, norm.T[1])) / 2

                color_ = (np.array(colors["diffuseColor"]) * 255).astype(np.int32)
                for p0 in points.astype(np.int32):
                    area1 = GL.triangle_area(p0, p[0, 0:2], p[1, 0:2])
                    area2 = GL.triangle_area(p0, p[1, 0:2], p[2, 0:2])
                    k = np.array([area1, area2, area - area1 - area2]) / area
                    z_ = 1.0 / np.dot(k, np.divide(1.0, np.abs(z))).sum()

                    # print(True, base_color)
                    # print(True, k, z_)
                    color_ = (z_ * ((k / np.abs(z)) @ base_color) * 255).astype(np.int32)
                    # print(True, color_)

                    gpu.GPU.draw_pixel(p0.tolist(), gpu.GPU.RGB8, color_)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
