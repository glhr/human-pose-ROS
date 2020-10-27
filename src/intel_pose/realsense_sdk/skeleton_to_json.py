import json
from vision_utils.logger import get_logger
logger=get_logger()

class SkeletonKeypoints:
    def __init__(self, joints, confidences, id):
        self.joints = joints
        self.confidences = confidences
        self.id = id
class Coordinate:
    def __init__(self, x,y):
        self.x, self.y = x,y

skeleton = [
SkeletonKeypoints(
joints=[
    Coordinate(x=-1.0, y=-1.0),
    Coordinate(x=653.1378173828125, y=296.25),
    Coordinate(x=735.7184448242188, y=288.75),
    Coordinate(x=773.255126953125, y=416.25),
    Coordinate(x=773.255126953125, y=498.75),
    Coordinate(x=570.5571899414062, y=311.25),
    Coordinate(x=578.0645141601562, y=558.75),
    Coordinate(x=630.6158447265625, y=716.25),
    Coordinate(x=728.2111206054688, y=521.25),
    Coordinate(x=728.2111206054688, y=716.25),
    Coordinate(x=-1.0, y=-1.0),
    Coordinate(x=630.6158447265625, y=536.25),
    Coordinate(x=653.1378173828125, y=716.25),
    Coordinate(x=-1.0, y=-1.0),
    Coordinate(x=-1.0, y=-1.0),
    Coordinate(x=-1.0, y=-1.0),
    Coordinate(x=668.1524658203125, y=198.75),
    Coordinate(x=608.0938110351562, y=206.25)],
confidences=[0.0, 0.8825716972351074, 0.7982165217399597, 0.9019602537155151, 0.7152760028839111, 0.8948974609375, 0.2098979651927948, 0.10448460280895233, 0.7081050872802734, 0.7135778665542603, 0.0, 0.7400355339050293, 0.5504853129386902, 0.0, 0.0, 0.0, 0.971623420715332, 0.21982541680335999],
id=0), SkeletonKeypoints(joints=[Coordinate(x=1148.6217041015625, y=191.25), Coordinate(x=1133.6070556640625, y=258.75), Coordinate(x=1073.54833984375, y=251.25), Coordinate(x=1013.48974609375, y=348.75), Coordinate(x=1028.50439453125, y=318.75), Coordinate(x=1201.1729736328125, y=273.75), Coordinate(x=1223.695068359375, y=386.25), Coordinate(x=1201.1729736328125, y=483.75), Coordinate(x=1043.51904296875, y=453.75), Coordinate(x=990.9677124023438, y=573.75), Coordinate(x=990.9677124023438, y=716.25), Coordinate(x=1126.0997314453125, y=476.25), Coordinate(x=1088.56298828125, y=618.75), Coordinate(x=1081.0556640625, y=716.25), Coordinate(x=1133.6070556640625, y=176.25), Coordinate(x=1163.6363525390625, y=176.25), Coordinate(x=1111.0850830078125, y=176.25), Coordinate(x=1186.1583251953125, y=183.75)], confidences=[0.9169151186943054, 0.8666443228721619, 0.8882757425308228, 0.8619902729988098, 0.7070491313934326, 0.9207891225814819, 0.9052256345748901, 0.8704547882080078, 0.8521599769592285, 0.8886606097221375, 0.4278188645839691, 0.8335491418838501, 0.7113293409347534, 0.2376931756734848, 0.910731315612793, 0.9182634353637695, 0.9543882608413696, 0.7800157070159912], id=1), SkeletonKeypoints(joints=[Coordinate(x=795.777099609375, y=378.75), Coordinate(x=840.8211059570312, y=393.75), Coordinate(x=833.3137817382812, y=393.75), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=863.3430786132812, y=401.25), Coordinate(x=833.3137817382812, y=468.75), Coordinate(x=780.762451171875, y=506.25), Coordinate(x=840.8211059570312, y=506.25), Coordinate(x=780.762451171875, y=551.25), Coordinate(x=773.255126953125, y=716.25), Coordinate(x=885.8651123046875, y=521.25), Coordinate(x=795.777099609375, y=551.25), Coordinate(x=795.777099609375, y=671.25), Coordinate(x=788.269775390625, y=371.25), Coordinate(x=803.2844848632812, y=363.75), Coordinate(x=-1.0, y=-1.0), Coordinate(x=833.3137817382812, y=348.75)], confidences=[0.7756341695785522, 0.7882691621780396, 0.5544993281364441, 0.0, 0.0, 0.9465464353561401, 0.707396924495697, 0.7032232880592346, 0.5784232020378113, 0.35206127166748047, 0.17989718914031982, 0.720065176486969, 0.6011240482330322, 0.35064825415611267, 0.6260064244270325, 0.8222935199737549, 0.0, 0.8321029543876648], id=2), SkeletonKeypoints(joints=[Coordinate(x=427.9178771972656, y=303.75), Coordinate(x=397.8885498046875, y=416.25), Coordinate(x=232.72727966308594, y=446.25), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=548.0352172851562, y=393.75), Coordinate(x=570.5571899414062, y=438.75), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=-1.0, y=-1.0), Coordinate(x=382.8739013671875, y=266.25), Coordinate(x=457.94720458984375, y=258.75), Coordinate(x=300.2932434082031, y=236.25), Coordinate(x=502.9912109375, y=243.75)], confidences=[0.7195408344268799, 0.503330647945404, 0.3491041958332062, 0.0, 0.0, 0.40657126903533936, 0.2615065574645996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6958167552947998, 0.9100594520568848, 0.8172404766082764, 0.4943097233772278], id=3)]

json_out = []
for person in skeleton:
    keypoints = []
    for i,kp in enumerate(person.joints):
        keypoints.extend([kp.x, kp.y, person.confidences[i]])
    json_out.append({'keypoints':keypoints})
print(json_out)

img_name = "test.png"
json_out_name = '../eval/realsense_sdk/' + img_name + '.predictions.json'
with open(json_out_name, 'w') as f:
    json.dump(json_out, f)
logger.info(json_out_name)
