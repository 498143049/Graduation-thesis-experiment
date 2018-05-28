import math
import numpy as np

def _mydesk(rad,w,h):
    crad = math.ceil(rad - 0.5)
    [x, y] = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(abs(x), abs(y))
    minxy = np.minimum(abs(x), abs(y))
    m1 = ((maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2 > rad * rad) * (minxy - 0.5) \
         + (((maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2 <= rad * rad) * np.lib.scimath.sqrt(
        rad * rad - (maxxy + 0.5) ** 2)).astype('float')

    m2 = (rad * rad > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + \
         ((rad * rad <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * \
          np.lib.scimath.sqrt(rad * rad - (maxxy - 0.5) ** 2))

    sgrid = (rad * rad * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) +
                          0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) -
             (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) \
            * ((((rad * rad < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) &
                 (rad * rad > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)) |
                ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad))))
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad * rad)
    sgrid[crad, crad] = min(np.pi * rad * rad, np.pi / 2)
    if ((crad > 0) and (rad > crad - 0.5) and (rad * rad < (crad - 0.5) * (crad - 0.5) + 0.25)):
        m1 = np.sqrt(rad * rad - (crad - 0.5) ** 2);
        m1n = m1 / rad;
        sg0 = 2 * (rad * rad * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5));
        sgrid[2 * crad, crad] = sg0;
        sgrid[crad, 2 * crad] = sg0;
        sgrid[crad, 0] = sg0;
        sgrid[0, crad] = sg0;
        sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0;
        sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0;
        sgrid[crad, 1] = sgrid[crad, 1] - sg0;
        sgrid[1, crad] = sgrid[1, crad] - sg0;
    # print(sgrid)
    sgrid[crad, crad] = np.minimum(sgrid[crad, crad], 1);
    if rad == h :
        dif = h-w
        sgrid = sgrid[:,dif:-dif]
    elif rad==w:
        dif = w - h
        sgrid = sgrid[ dif:-dif, :]
    h = sgrid / np.sum(sgrid);
    return h,sgrid;

# kp = [(x,y) for y in range(0, 2, 1)
#           for x in range(0, 3, 1)]
# print(kp)

def getmax_min(x,y):
    if x>y:
        return x,y
    else:
        return y,x
def get_dsavlue(x1, y1, zxp, zyp):
    if(math.abs(x1[0] - y1[0])>2 or math.abs(x1[1] - y1[1])>2):
        return 0;
    else :
        # 通过矩阵分解求得其值
        max_x, min_x =  getmax_min(x1[0], y1[0]);
        max_y, min_y =  getmax_min(x1[1], y1[1]);
        gx = (zxp[min_x - 1: max_x + 1][min_y - 1: max_y + 1]) * K
        gy = (zyp[min_x - 1: max_x + 11][min_y - 1: max_y + 1]) * K
        G = np.array([gx.flatten(1), gy.flatten(1)]).T
        u, s, v = np.linalg.svd(G, full_matrices=False)
        v = v.T
        S1 = (s[0] + 1.0) / (s[1] + 1.0)
        S2 = 1.0 / S1
        m1 = (S1 * v[:, 0])
        m2 = v[:, 0]
        m3 = np.dot(np.array([m1]).T, np.array([m2]))
        m4 = (S2 * v[:, 1])
        m5 = v[:, 1]
        m6 = np.dot(np.array([m4]).T, np.array([m5]))
        m7 = ((s[0] * s[1] + 0.0000001) / le) ** alpha
        tmp = (m3 + m6) * m7
        #  得到矩阵分解的值
