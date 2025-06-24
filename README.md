Repository: daniyalse/brain-tumor-segmentation-yolo-sam
Files analyzed: 1000

Estimated tokens: 333.3k

Directory structure:
└── daniyalse-brain-tumor-segmentation-yolo-sam/
    ├── READEME.md.txt
    ├── yolo11n.pt
    ├── Masks/
    ├── runs/
    ├── TestRuns/
    ├── TumorDetection/
    │   ├── data.yaml
    │   ├── README.roboflow.txt
    │   ├── test/
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   │       ├── glioma_102_jpg.rf.e184590079f7726ff64daccd83d8ff99.txt
    │   │       ├── glioma_1086_jpg.rf.a2785a388f7efbd5f665601ec9147d36.txt
    │   │       ├── glioma_1112_jpg.rf.8f37ed9f604563cb38cb6f32ce3acf16.txt
    │   │       ├── glioma_1133_jpg.rf.8f564b9043b1ec8362fc38df344bbe1e.txt
    │   │       ├── glioma_1138_jpg.rf.c6a259e4343f9bfb52cfa69eb60f46c7.txt
    │   │       ├── glioma_1147_jpg.rf.e1dfee64ef3faccf1c0ace058d230551.txt
    │   │       ├── glioma_1202_jpg.rf.24044ec3eab86394cd061a4a8735a0d9.txt
    │   │       ├── glioma_120_jpg.rf.7103603cabccecd5d6c0a7d875b2b5d7.txt
    │   │       ├── glioma_1228_jpg.rf.068c264838666499bca0d7060de33af9.txt
    │   │       ├── glioma_1235_jpg.rf.2a8808fa4ef7eeca9ee3b039dd233373.txt
    │   │       ├── glioma_1240_jpg.rf.f9ce263457e8d2b30da8d2876f8ac132.txt
    │   │       ├── glioma_1266_jpg.rf.8d9c69dba00e64fcf859ffe1b8f34ed7.txt
    │   │       ├── glioma_1272_jpg.rf.634083758e76317aecd2f06cf736ebb4.txt
    │   │       ├── glioma_128_jpg.rf.b238fdefc66ce62f90c11aa4ff7cd10f.txt
    │   │       ├── glioma_1296_jpg.rf.c0e3188a4d00bfba320b904486a84794.txt
    │   │       ├── glioma_1301_jpg.rf.d70f5b0990be3d5ca5d2359623d84a90.txt
    │   │       ├── glioma_145_jpg.rf.9125a688625012628983f59e080c1986.txt
    │   │       ├── glioma_171_jpg.rf.ca11a72a9de2a480ff956b4306c2ea45.txt
    │   │       ├── glioma_172_jpg.rf.2456b08df0d238e939af32d2a0f31ae8.txt
    │   │       ├── glioma_191_jpg.rf.77be681637a02f58673ed32218116004.txt
    │   │       ├── glioma_22_jpg.rf.991216b0b4606e74817673a45b166a34.txt
    │   │       ├── glioma_238_jpg.rf.51666b7b153f1e603cf1558cf99f7e7d.txt
    │   │       ├── glioma_259_jpg.rf.60db168e3d887d8db6795dd989766381.txt
    │   │       ├── glioma_283_jpg.rf.d2e24030d360608bbb0038c7073d7faf.txt
    │   │       ├── glioma_31_jpg.rf.f88f3b4e4a83554f8bde206b1b9bce4f.txt
    │   │       ├── glioma_322_jpg.rf.0fa79c110072428a02bb441c57901d3f.txt
    │   │       ├── glioma_358_jpg.rf.8cd4754751b64828d47510309613f1a0.txt
    │   │       ├── glioma_374_jpg.rf.cd5b6cddcb663677f866adeafb8448a5.txt
    │   │       ├── glioma_392_jpg.rf.b7aa80b4d570d52b3122bbafb7961b86.txt
    │   │       ├── glioma_414_jpg.rf.f9888de8bd4c5d90586fb849d7d933ed.txt
    │   │       ├── glioma_468_jpg.rf.d85e769874f88b31113296c64243faa6.txt
    │   │       ├── glioma_474_jpg.rf.9a407f6530fa06506f37d610565362e1.txt
    │   │       ├── glioma_511_jpg.rf.f84ee5cf0c0f3fc5f6af758d55fdfd58.txt
    │   │       ├── glioma_51_jpg.rf.d45bc240398f7a69aa84fc12190fcffc.txt
    │   │       ├── glioma_522_jpg.rf.f5d4d387b72a07daeecf462fe1f4a151.txt
    │   │       ├── glioma_525_jpg.rf.5fc207ce06a7678e3ddffb00c411105a.txt
    │   │       ├── glioma_543_jpg.rf.5e7686edfc83b46e7eddc407a8649497.txt
    │   │       ├── glioma_550_jpg.rf.b52127c0a77a8785f44d94766365a326.txt
    │   │       ├── glioma_551_jpg.rf.aac527f5cb1dcfdfa00a5486b61b973b.txt
    │   │       ├── glioma_578_jpg.rf.54b68964d2a5408bbd1bba9919a4488a.txt
    │   │       ├── glioma_609_jpg.rf.e564f70d7e6adff47de4b52b92e46753.txt
    │   │       ├── glioma_629_jpg.rf.563773926f1e45d0ef9ac57c3cfad830.txt
    │   │       ├── glioma_650_jpg.rf.10b1beda8b03d28b0450af5be9c0b215.txt
    │   │       ├── glioma_665_jpg.rf.c8f631906a0210b39ce82d1f6ee531d9.txt
    │   │       ├── glioma_666_jpg.rf.6e859f6183e409c261f3e872632c0fce.txt
    │   │       ├── glioma_70_jpg.rf.c053d9e585343a2d87980ac48b2b827a.txt
    │   │       ├── glioma_722_jpg.rf.0973288dd768103902dbadadb02ff063.txt
    │   │       ├── glioma_835_jpg.rf.7c101a77efb377960f7739ae8d47f4db.txt
    │   │       ├── glioma_904_jpg.rf.46fbdba7d5697ea773c86f0214380230.txt
    │   │       ├── glioma_906_jpg.rf.3b56fc6141078c119069f74ca103cd22.txt
    │   │       ├── glioma_930_jpg.rf.eb6881f20608f5482378962913095da7.txt
    │   │       ├── glioma_933_jpg.rf.6691f90d0b86b8a81e91b978f4d02543.txt
    │   │       ├── glioma_97_jpg.rf.1467ac8594412f669ecab1c802dd5b11.txt
    │   │       ├── meningioma_1003_jpg.rf.1cb046de3a4bc224cd1d63fb5fa5561f.txt
    │   │       ├── meningioma_101_jpg.rf.bbe32c57f14ff5362085ea9560be430b.txt
    │   │       ├── meningioma_1020_jpg.rf.c083f0e6892b08d794d09d0ad935bdd5.txt
    │   │       ├── meningioma_1030_jpg.rf.7a3c8c1da556e3207c3779e303396f0d.txt
    │   │       ├── meningioma_1031_jpg.rf.92d5d9e4f499a8616701629db1bbf8e6.txt
    │   │       ├── meningioma_1032_jpg.rf.6f42b84ac1d4bc6443068a2d2dfe4b84.txt
    │   │       ├── meningioma_1040_jpg.rf.d34a9c9b9fecc69c30ac22cda6a360d0.txt
    │   │       ├── meningioma_1048_jpg.rf.a4b0c3d62dc06cc46a19fa7457356dbf.txt
    │   │       ├── meningioma_1051_jpg.rf.a2b35bd71479146c9668177cf6dfa968.txt
    │   │       ├── meningioma_1052_jpg.rf.14993d2fea569cfcd8576a5561bba274.txt
    │   │       ├── meningioma_1053_jpg.rf.4f69cbaaa68a46b0b815f148e856f8fb.txt
    │   │       ├── meningioma_1067_jpg.rf.2876f08236a9221433638d0e01ceef85.txt
    │   │       ├── meningioma_1072_jpg.rf.a7c68a0d0357e08de8b940f3c6c69ed4.txt
    │   │       ├── meningioma_1076_jpg.rf.4fa57284674e5fbf07d6446ff67d0a76.txt
    │   │       ├── meningioma_1079_jpg.rf.802709b4226820338bef07534cabe5b6.txt
    │   │       ├── meningioma_1083_jpg.rf.2f6bc4256771a58ea2c38f4575492a37.txt
    │   │       ├── meningioma_1085_jpg.rf.4b6969bbe6585cddc76c5ff75db16419.txt
    │   │       ├── meningioma_1087_jpg.rf.ec345f0d4e2591a385d025fd56355338.txt
    │   │       ├── meningioma_1098_jpg.rf.bad8fc08770434b25e8f030f21da68bc.txt
    │   │       ├── meningioma_1100_jpg.rf.44e77aef5e60423e99363a50dde7c098.txt
    │   │       ├── meningioma_1103_jpg.rf.8f9a1d7d54389a3e56587b3abd7e4a36.txt
    │   │       ├── meningioma_1118_jpg.rf.00df375c7507bc402d95140278e74d13.txt
    │   │       ├── meningioma_1128_jpg.rf.33c271ae0482d8d0c866b985152e5d7c.txt
    │   │       ├── meningioma_1134_jpg.rf.d1abfc3ea3ca3e76d4889b36dad0d49a.txt
    │   │       ├── meningioma_113_jpg.rf.d837df7cfe8a79f90887240e07a400bc.txt
    │   │       ├── meningioma_1147_jpg.rf.48ec0ae1b9ac50e1614190b9198cfe29.txt
    │   │       ├── meningioma_1153_jpg.rf.cd5615803871925f83c0420197f243ba.txt
    │   │       ├── meningioma_1166_jpg.rf.57f037b5856750e964733a2977579ea6.txt
    │   │       ├── meningioma_1175_jpg.rf.a4b25fbfaa57544d4497a754fca96349.txt
    │   │       ├── meningioma_1178_jpg.rf.d28ae433a62770aae96bb76711418137.txt
    │   │       ├── meningioma_117_jpg.rf.77f8453394784a23e30b31d223402581.txt
    │   │       ├── meningioma_1183_jpg.rf.057d898ba6a98d81499ebed48dd2092c.txt
    │   │       ├── meningioma_1185_jpg.rf.af26d6157a2ec8fdb58a10c755004c7b.txt
    │   │       ├── meningioma_1195_jpg.rf.baa094c6182811a33bf00b56230f0076.txt
    │   │       ├── meningioma_1200_jpg.rf.8f1194582c8142887ff4537a9e41c294.txt
    │   │       ├── meningioma_1201_jpg.rf.4155c14224b63c3a4bbf6d570a2a4089.txt
    │   │       ├── meningioma_120_jpg.rf.f86ba2e7977ba292b26c91f2564d946a.txt
    │   │       ├── meningioma_1210_jpg.rf.9ae7dc5b727491b6603d19416676a7c8.txt
    │   │       ├── meningioma_1212_jpg.rf.72220ada901dcb13a50fcfd9483ee956.txt
    │   │       ├── meningioma_1220_jpg.rf.163bb25095405d38a11f58b0fb559c0b.txt
    │   │       ├── meningioma_1233_jpg.rf.d62273e91257dbcb25bdaac2b1cd42af.txt
    │   │       ├── meningioma_1234_jpg.rf.0c1c2a6eb11bffeb99e313f5685f9b87.txt
    │   │       ├── meningioma_1235_jpg.rf.6d4ea4c1965d6fbaeac495438bc29a62.txt
    │   │       ├── meningioma_1242_jpg.rf.ddf04a2d94430995ea1db196c804f82f.txt
    │   │       ├── meningioma_1243_jpg.rf.0024eff03061b426a5f4a93cc1b539cd.txt
    │   │       ├── meningioma_1249_jpg.rf.5ae9d159cbf1096b85e4ab138757395f.txt
    │   │       ├── meningioma_1250_jpg.rf.a13f606fa448cebff1712b9bf916612d.txt
    │   │       ├── meningioma_1255_jpg.rf.f1ab0ba8c39a1ba9c030649227ad01f4.txt
    │   │       ├── meningioma_1256_jpg.rf.9b22b2792f72a75a8a7b76642ee9d16b.txt
    │   │       ├── meningioma_1269_jpg.rf.ba78521135708aa900973fc31f1144e9.txt
    │   │       ├── meningioma_1280_jpg.rf.1ed02c7f20cb4cbcd709e7e327e36016.txt
    │   │       ├── meningioma_1281_jpg.rf.156dc24e062ef1db758b7923891a7474.txt
    │   │       ├── meningioma_1282_jpg.rf.99fa3552b33841ac55cc812908286092.txt
    │   │       ├── meningioma_1297_jpg.rf.a9a4f7f04148883e328de995b2b18a61.txt
    │   │       ├── meningioma_129_jpg.rf.02cf3f9072386c01e700156af5c8f3a1.txt
    │   │       ├── meningioma_1300_jpg.rf.1205de8340edfbe7473552dc47010138.txt
    │   │       ├── meningioma_1307_jpg.rf.1562868b20f5f63c07ec0c9eb9121700.txt
    │   │       ├── meningioma_1318_jpg.rf.c6a8e0e52fae3f2a948ece1d1763f2e9.txt
    │   │       ├── meningioma_1319_jpg.rf.51b10a74243da6bcaa41ebce7532006a.txt
    │   │       ├── meningioma_1323_jpg.rf.edfca5eda6a7b4f9f60ff05875a85e07.txt
    │   │       ├── meningioma_1336_jpg.rf.7f86928168516cd98a36481db162a727.txt
    │   │       ├── meningioma_134_jpg.rf.974249c3aefbd2743a3134e9f4340fe8.txt
    │   │       ├── meningioma_136_jpg.rf.8491175983470b61181aad86eb939f8d.txt
    │   │       ├── meningioma_138_jpg.rf.6b7cc548f3be2b200133aaaf73d5a989.txt
    │   │       ├── meningioma_139_jpg.rf.dadc43b47996c23ec8abd05dd8a078aa.txt
    │   │       ├── meningioma_141_jpg.rf.37375ab0537220a06979363c6e8b436f.txt
    │   │       ├── meningioma_147_jpg.rf.1bbf03ae4cebe580fa339f3085bf854b.txt
    │   │       ├── meningioma_148_jpg.rf.9c6edd3b213eac261b1f461446504426.txt
    │   │       ├── meningioma_155_jpg.rf.acfb1ac57a47166618309233d3e09e69.txt
    │   │       ├── meningioma_158_jpg.rf.3b1a3d322fc03483cb5d55d34fcd06fe.txt
    │   │       ├── meningioma_161_jpg.rf.508b6ec1ca195f8c4cf6e18d2231c18a.txt
    │   │       ├── meningioma_168_jpg.rf.635b9d1dafc64618f2c47d441d871c9c.txt
    │   │       ├── meningioma_169_jpg.rf.3496fb138a70dfd25edb4dbb8e71707b.txt
    │   │       ├── meningioma_174_jpg.rf.c253ac6b03709518b654cf1b846071f9.txt
    │   │       ├── meningioma_185_jpg.rf.b5013a5b232b32fe0e1daa6e17db4db2.txt
    │   │       ├── meningioma_187_jpg.rf.63e58c7b1736a9c3757b1e9189626f4d.txt
    │   │       ├── meningioma_189_jpg.rf.41e99170113b9892ffc42d2040231ad3.txt
    │   │       ├── meningioma_190_jpg.rf.906cfd2d2b77ca580dfa533136bfd3d6.txt
    │   │       ├── meningioma_195_jpg.rf.5da508c5656d120630a308e8b43e8f31.txt
    │   │       ├── meningioma_196_jpg.rf.0d1f270cff963f8af36775e06432849e.txt
    │   │       ├── meningioma_201_jpg.rf.409fa0316610f1968bcde41bd4fdde82.txt
    │   │       ├── meningioma_203_jpg.rf.ea11dc83894c1cd9262ca4d2f1854c40.txt
    │   │       ├── meningioma_206_jpg.rf.9882ab4219a33378e9c675ce3bbec31e.txt
    │   │       ├── meningioma_222_jpg.rf.bf76ef8786bdbe5ab070a12f63e910ec.txt
    │   │       ├── meningioma_223_jpg.rf.7755bf1b15746ddda3a67acee86065a4.txt
    │   │       ├── meningioma_225_jpg.rf.9522022bbfd91e2a307e855ac319cf27.txt
    │   │       ├── meningioma_234_jpg.rf.0915b44eb766c431160ad46afd99459b.txt
    │   │       ├── meningioma_242_jpg.rf.bb47982bb3a184a34afaa4af3f515297.txt
    │   │       ├── meningioma_245_jpg.rf.54a22411ef95106132d6b451b999f237.txt
    │   │       ├── meningioma_25_jpg.rf.3859c503774a53bb0c6b5b9349562c6a.txt
    │   │       ├── meningioma_263_jpg.rf.70ea095866132fa8c4c048652b93160d.txt
    │   │       ├── meningioma_269_jpg.rf.f4edec265e7781710c266b9f18689699.txt
    │   │       ├── meningioma_273_jpg.rf.4f3ebc5e793c8df72fa7e3f2a6ab4300.txt
    │   │       ├── meningioma_280_jpg.rf.d16ce0cd86f7e265c2855bcec0ba02dc.txt
    │   │       ├── meningioma_285_jpg.rf.7ca8f304a112af21e3be1a85ff7013c2.txt
    │   │       ├── meningioma_288_jpg.rf.b74605bf97d174a49dd10f4b8eac6f61.txt
    │   │       ├── meningioma_294_jpg.rf.00a8bb2e998110e2e598b32410f39c36.txt
    │   │       ├── meningioma_295_jpg.rf.ecfaa49157a1d1af65a80a1e4f3de438.txt
    │   │       ├── meningioma_308_jpg.rf.69600117582e60b8e7cf3652c3f8723c.txt
    │   │       ├── meningioma_310_jpg.rf.324dc73989291cfcacf2f7ff6f1f9b71.txt
    │   │       ├── meningioma_312_jpg.rf.556f795ceb5e9714556ad461c6ee13cc.txt
    │   │       ├── meningioma_316_jpg.rf.0c8dc01bd2d691310062457ae79ff838.txt
    │   │       ├── meningioma_318_jpg.rf.497f2dcaefcfdfe088110cffe3a48de1.txt
    │   │       ├── meningioma_319_jpg.rf.9cc88ece353c5431b9f49d7ed6c83ad1.txt
    │   │       ├── meningioma_325_jpg.rf.cecf06691481e910dde208a823f0d390.txt
    │   │       ├── meningioma_331_jpg.rf.c123e69b071dd856fd282204f9ec4e36.txt
    │   │       ├── meningioma_342_jpg.rf.30a0372c2bafc1ec4a44fb86c84c1595.txt
    │   │       ├── meningioma_343_jpg.rf.8088aa8dac38e2d92a21c145367ed44f.txt
    │   │       ├── meningioma_346_jpg.rf.af45eb1c56f185c1b4fce658d886cb91.txt
    │   │       ├── meningioma_347_jpg.rf.86020889954b4fd005848f8754283ac6.txt
    │   │       ├── meningioma_355_jpg.rf.239a9f8e75698f9fbc93ab2fc445a9e0.txt
    │   │       ├── meningioma_35_jpg.rf.7c99aa29547ebcb3a8cb8a78c95ad76a.txt
    │   │       ├── meningioma_368_jpg.rf.9d74c6881f81c543b1c9a5cef74a6537.txt
    │   │       ├── meningioma_371_jpg.rf.6ed100fc09c2a6066a7a698a7a861797.txt
    │   │       ├── meningioma_375_jpg.rf.380d15275cfa0e4d376525ae9c86a507.txt
    │   │       ├── meningioma_384_jpg.rf.1156717de547acc9888b45144028e2f3.txt
    │   │       ├── meningioma_386_jpg.rf.837a2af568f580f346a797f1bd5015cc.txt
    │   │       ├── meningioma_391_jpg.rf.c05273c79dfe211092dbc147bc57f96c.txt
    │   │       ├── meningioma_392_jpg.rf.a7e384c4a7b1146a935811a2c04e47b4.txt
    │   │       ├── meningioma_398_jpg.rf.f303a581bba88e387c4a1ab8dbbdd418.txt
    │   │       ├── meningioma_3_jpg.rf.3d6ab426da351985af9403198a44e6f0.txt
    │   │       ├── meningioma_401_jpg.rf.e82a83f493a3fdbb2c7f4ec1fb0c9397.txt
    │   │       ├── meningioma_410_jpg.rf.ff44de7d5514fa24f683f3fbdc2ddea2.txt
    │   │       ├── meningioma_412_jpg.rf.e9cb10656771b59ce7c885fd2bfb6665.txt
    │   │       ├── meningioma_415_jpg.rf.8960d1e0deb57ffc78750398cb7410f5.txt
    │   │       ├── meningioma_424_jpg.rf.6e971bda062dbe09e62cc3939de38b3f.txt
    │   │       ├── meningioma_427_jpg.rf.912d9eb73f6ae6d1019c74b3b8354217.txt
    │   │       ├── meningioma_428_jpg.rf.161b483f1c43282a19796f3db11ba607.txt
    │   │       ├── meningioma_446_jpg.rf.586ea523102745fbd6d7d19556a4f4a9.txt
    │   │       ├── meningioma_449_jpg.rf.8add6f9e9b0a0427a42f614fa15f4873.txt
    │   │       ├── meningioma_460_jpg.rf.0faac894ebf5dec9e5112ced6dc5bbb2.txt
    │   │       ├── meningioma_466_jpg.rf.59370219c28541ae27c863a2a39e37d3.txt
    │   │       ├── meningioma_46_jpg.rf.259eebd958cd27bed31605e0c64f2f20.txt
    │   │       ├── meningioma_473_jpg.rf.064ae57ceafe17ae32c5e4b6792bac84.txt
    │   │       ├── meningioma_474_jpg.rf.3489d5edbb9f106b58990bec025051c4.txt
    │   │       ├── meningioma_477_jpg.rf.44cfb7a6c54822440200eded86616949.txt
    │   │       ├── meningioma_482_jpg.rf.04fa49e7085ab5dde5a0a4a1bc568640.txt
    │   │       ├── meningioma_484_jpg.rf.b0dacc7c4c7d73ebaf772cea4de6c25f.txt
    │   │       ├── meningioma_489_jpg.rf.37968ddddd2f75afc081ab00dde4b5f7.txt
    │   │       ├── meningioma_502_jpg.rf.c6df0ca015dbbe0d5e7633f9c0d182ee.txt
    │   │       ├── meningioma_513_jpg.rf.9ccd3e3625158e49d46d039ad5808006.txt
    │   │       ├── meningioma_517_jpg.rf.10c6b3b1eaec17a29c6747d5d7ff1d64.txt
    │   │       ├── meningioma_518_jpg.rf.299c80c7b531ee4bc9516f9d56b9d8a6.txt
    │   │       ├── meningioma_520_jpg.rf.7fbfefe2feacebd4d94756793fc247c8.txt
    │   │       ├── meningioma_523_jpg.rf.517e6cbcb701c14604ae34affef7c00c.txt
    │   │       ├── meningioma_526_jpg.rf.98e109530bc9b3ede3d6e6cd4a06050e.txt
    │   │       ├── meningioma_540_jpg.rf.c7dba9ffd13a4576620ef3aeb2b60ea8.txt
    │   │       ├── meningioma_543_jpg.rf.e11bae3ffd29e3d4d0394811c4dab3d3.txt
    │   │       ├── meningioma_544_jpg.rf.4f027eb4b90a9d8ce0b3d0370b2d0ee8.txt
    │   │       ├── meningioma_547_jpg.rf.ce87498c81d59d6499864d02df593180.txt
    │   │       ├── meningioma_55_jpg.rf.ef13fdd7d8b8b5c722d124c14792f38c.txt
    │   │       ├── meningioma_561_jpg.rf.544bf3fac642cba5ff23b9ec6f8fa39d.txt
    │   │       ├── meningioma_564_jpg.rf.b6b6cf99462359a70869fe1bb7fa9fa6.txt
    │   │       ├── meningioma_567_jpg.rf.a077d99c54ffc0d725d9c939a64ce443.txt
    │   │       ├── meningioma_56_jpg.rf.809e46b330e6d1e7c74a8b41330f8f6b.txt
    │   │       ├── meningioma_584_jpg.rf.3d0ed1863e116fea7147a0411f1bab22.txt
    │   │       ├── meningioma_585_jpg.rf.5df72b5d661e615e520bf19ed05752f1.txt
    │   │       ├── meningioma_588_jpg.rf.2b39f2bd67146ad3088d621a41d0a05e.txt
    │   │       ├── meningioma_602_jpg.rf.b3e6e4fd5cead908876400224522aa10.txt
    │   │       ├── meningioma_61_jpg.rf.2ccd18e873063f2ed8d54852e9f66835.txt
    │   │       ├── meningioma_624_jpg.rf.cbc39b6ccc13403b879e6bd5da8ac1c8.txt
    │   │       ├── meningioma_62_jpg.rf.23ac77d0e5264c24c31a81aaffe72da1.txt
    │   │       ├── meningioma_638_jpg.rf.63183096ef023fd6179343f832e5b918.txt
    │   │       ├── meningioma_63_jpg.rf.09d8adece431eaa739db4cd3fa47f681.txt
    │   │       ├── meningioma_64_jpg.rf.b86f2a96ab8c4293e2ceb0fdbb133b31.txt
    │   │       ├── meningioma_654_jpg.rf.404dce2eaf6f78e861d07817c3b35726.txt
    │   │       ├── meningioma_656_jpg.rf.8dd8eac691401cc5e8034bcdb6c17947.txt
    │   │       ├── meningioma_657_jpg.rf.f7be9f7383153fc39c77db5d4cc6d494.txt
    │   │       ├── meningioma_662_jpg.rf.d4f24820353a5f90136f3af5fc78abe1.txt
    │   │       ├── meningioma_664_jpg.rf.ff9460f4f6c62598a5aee99e4d8c37fb.txt
    │   │       ├── meningioma_677_jpg.rf.533e48c81a54f8bb29c2015e3b47fdad.txt
    │   │       ├── meningioma_689_jpg.rf.f4110e21ea8ba8fdd8b2a45a6127a7c6.txt
    │   │       ├── meningioma_693_jpg.rf.8d77500198a65eae38bcf876f27db18a.txt
    │   │       ├── meningioma_695_jpg.rf.e887b2b087c43c663ead6aeb47cdd2cd.txt
    │   │       ├── meningioma_707_jpg.rf.45ba1e9cd6b073aacc11936ced9d3564.txt
    │   │       ├── meningioma_710_jpg.rf.1bba5ad4daee6917d9520f832a0986db.txt
    │   │       ├── meningioma_711_jpg.rf.924f1f4a9927e0df7cdb4c153f1807a3.txt
    │   │       ├── meningioma_733_jpg.rf.3df47e74401e0b8871f177d0087ceef2.txt
    │   │       ├── meningioma_734_jpg.rf.79af18bc94491cc52fb5a9aefdf4b086.txt
    │   │       ├── meningioma_73_jpg.rf.737b7d02c5566539f8c6e99c61f1bb42.txt
    │   │       ├── meningioma_748_jpg.rf.589414d2dd21bdf2b0752f6fa1959865.txt
    │   │       ├── meningioma_749_jpg.rf.deaf1e32a98b5b9178da63bfdfd3adcd.txt
    │   │       ├── meningioma_756_jpg.rf.d642735914e417b8a9c80818bd8cd2e1.txt
    │   │       ├── meningioma_757_jpg.rf.ba3c9be419657e1322b9c5643a89b7cf.txt
    │   │       ├── meningioma_758_jpg.rf.773269389e2b3a0c0ed1bfb45ef85194.txt
    │   │       ├── meningioma_759_jpg.rf.b186884d401c9c2a145b3edbdb9cbc6b.txt
    │   │       ├── meningioma_773_jpg.rf.c5ae9781857e9a783562dae0412f873b.txt
    │   │       ├── meningioma_776_jpg.rf.1987625f340e41a7ef8d410a4c5345bd.txt
    │   │       ├── meningioma_779_jpg.rf.1e1082dfcadaedbd005b5cfc6e06cef2.txt
    │   │       ├── meningioma_781_jpg.rf.9d44a2d15a217600ca07c83785ab7d78.txt
    │   │       ├── meningioma_782_jpg.rf.f20f3818a42dad9ea5ba2054824bfca2.txt
    │   │       ├── meningioma_787_jpg.rf.b4fc7fd76dd101524b5e2c01bc4a0dfc.txt
    │   │       ├── meningioma_792_jpg.rf.5cd51b220e3a65606c72bd389a656e38.txt
    │   │       ├── meningioma_793_jpg.rf.2b1cddb8721fff606bf90a695ccdf926.txt
    │   │       ├── meningioma_794_jpg.rf.e6ea105c447a297c2c57171da676b416.txt
    │   │       ├── meningioma_797_jpg.rf.23361b2204143a425ec4e7e6404ce7e2.txt
    │   │       ├── meningioma_800_jpg.rf.4f2b2bb0491e2960a62f39eb8778e6c0.txt
    │   │       ├── meningioma_801_jpg.rf.30da0e61ac1939f9b4fe7cf248e7c52a.txt
    │   │       ├── meningioma_817_jpg.rf.e627f4d8fe10890aa4f608d3811e4590.txt
    │   │       ├── meningioma_818_jpg.rf.d0a0db73578541a04e423e37c4d5cd79.txt
    │   │       ├── meningioma_826_jpg.rf.076b51e8b7ba356f11c7b349609c5525.txt
    │   │       ├── meningioma_827_jpg.rf.670af4536eb3590bcdb1da7671ee42ca.txt
    │   │       ├── meningioma_842_jpg.rf.00a12ad362ae2288f7cde416c23d8398.txt
    │   │       ├── meningioma_851_jpg.rf.5cb883da6643e552e4b05b22f1f665de.txt
    │   │       ├── meningioma_864_jpg.rf.dd99018be6b80c0479ba358d9a1e6987.txt
    │   │       ├── meningioma_866_jpg.rf.6eeaba9ea919ac847aacf7795c4bb3d7.txt
    │   │       ├── meningioma_867_jpg.rf.c0d07726c11b7a84a5fa4d1fc0a8505f.txt
    │   │       ├── meningioma_86_jpg.rf.bdd9bf4edaeff7e33264907091feeccd.txt
    │   │       ├── meningioma_872_jpg.rf.bcff08f393b9f39298578cc9457afffd.txt
    │   │       ├── meningioma_882_jpg.rf.62b9c653266173b287c6db075d91cba7.txt
    │   │       ├── meningioma_889_jpg.rf.08457415fb8ec9d28c77b345a9bd7d65.txt
    │   │       ├── meningioma_893_jpg.rf.ad34383d7ff8913bb6e72a1927545be6.txt
    │   │       ├── meningioma_896_jpg.rf.1088dce4543b6291543f15f3c9a5ef53.txt
    │   │       ├── meningioma_901_jpg.rf.7f26f443a85ace9eeb2e372adf5b737d.txt
    │   │       ├── meningioma_913_jpg.rf.8fa76a39c193b325ce78fe62c88ce2c2.txt
    │   │       ├── meningioma_91_jpg.rf.9376f3bcef6ca61b2d4f0dde0d35a9b0.txt
    │   │       ├── meningioma_924_jpg.rf.ef23b2a126ea5c99ead25e46dcbb67e6.txt
    │   │       ├── meningioma_935_jpg.rf.2820d1d475b4a80653760ac515bcf47f.txt
    │   │       ├── meningioma_937_jpg.rf.8da0c940919adb2db3045845d5a731e6.txt
    │   │       ├── meningioma_944_jpg.rf.86b8a982d384dfff1deeaa03e63e97db.txt
    │   │       ├── meningioma_958_jpg.rf.c45c2745072466a253c58e32f0360c49.txt
    │   │       ├── meningioma_95_jpg.rf.ed230bc3cdc5214e8911b1958cf547e3.txt
    │   │       ├── meningioma_962_jpg.rf.dc5529c452baf31a925796b88183ea0b.txt
    │   │       ├── meningioma_983_jpg.rf.d9b0a28a097dc6b2bb1de1e2a293f3a1.txt
    │   │       ├── meningioma_986_jpg.rf.bcacf1010002848f9f3dd0c9dd22fb3d.txt
    │   │       ├── meningioma_989_jpg.rf.d71f7461ce2b6395ac9695c8409cd1b1.txt
    │   │       ├── meningioma_992_jpg.rf.7ecd4a2363705b5217df4eb8f7b620e5.txt
    │   │       ├── meningioma_993_jpg.rf.4bf82291957b6780725ba736bfbb4ae0.txt
    │   │       ├── meningioma_99_jpg.rf.14fda29cfbc391267067f1a140776f9e.txt
    │   │       ├── no_tumor_1012_jpg.rf.b7971dea3b79562d4e76cd9c3e6bbad6.txt
    │   │       ├── no_tumor_1021_jpg.rf.61b4d9a8483a617a2cdd62ec51c14fad.txt
    │   │       ├── no_tumor_1039_jpg.rf.01d362e9945468a352a542a18c38ded1.txt
    │   │       ├── no_tumor_1041_jpg.rf.c2ad0c6c5b3237dcb37ed3bef1ba557e.txt
    │   │       ├── no_tumor_1045_jpg.rf.2634e2ce15ebbd1c21f43d70e8044f8f.txt
    │   │       ├── no_tumor_1050_jpg.rf.b94da538ccc63eb4220a4679c5b88f84.txt
    │   │       ├── no_tumor_1068_jpg.rf.2d663ba9dee8887033595f92974557c5.txt
    │   │       ├── no_tumor_1073_jpg.rf.5bc35117e317c026e72719bc078cbd8c.txt
    │   │       ├── no_tumor_1079_jpg.rf.ae610d05451e54ff6492a5e213ebd372.txt
    │   │       ├── no_tumor_1085_jpg.rf.c4e0c79ffc9f4a82cbf6a27f594bd1ce.txt
    │   │       ├── no_tumor_1088_jpg.rf.9a5310703a4cb1600585cfc4dae70d87.txt
    │   │       ├── no_tumor_1093_jpg.rf.aea4c307a9ae60e321de8879b8adae27.txt
    │   │       ├── no_tumor_1103_jpg.rf.f5d51689f9a518c267c2095c9ee98bf7.txt
    │   │       ├── no_tumor_1113_jpg.rf.152593fcacfb8267ecd4d1e1e23db8df.txt
    │   │       ├── no_tumor_1116_jpg.rf.58d57065c60fae62176fd8277ddfe6c0.txt
    │   │       ├── no_tumor_1122_jpg.rf.b62a4d7cc6933fd7161d203f8dd61875.txt
    │   │       ├── no_tumor_112_jpg.rf.afdc41cfb3f5fbb0c626fcff74f2d6e5.txt
    │   │       ├── no_tumor_1153_jpg.rf.44296b2708ffccf2650bd2dc305fc38f.txt
    │   │       ├── no_tumor_1157_jpg.rf.93f41da013b86c9dac838699ebbce758.txt
    │   │       ├── no_tumor_1210_jpg.rf.4ce02806ade4c1f448b354d2e6bc60cc.txt
    │   │       ├── no_tumor_1215_jpg.rf.5839678b4f833068a54ee551626d4cac.txt
    │   │       ├── no_tumor_1237_jpg.rf.2b4e1824fadb8f068b44eaa43a7ea4fe.txt
    │   │       ├── no_tumor_1251_jpg.rf.93d7f42c965eb13ed1d52349e5b56fb3.txt
    │   │       ├── no_tumor_1264_jpg.rf.3a4558975d3e7aa529cc9e050b1e1848.txt
    │   │       ├── no_tumor_1265_jpg.rf.7c655129f792360998c6cfc59a897dbd.txt
    │   │       ├── no_tumor_126_jpg.rf.50e10f563b8a940f1503c7dfb8cb6021.txt
    │   │       ├── no_tumor_1271_jpg.rf.99b32784706552b3695ed804e917a16e.txt
    │   │       ├── no_tumor_1278_jpg.rf.9034ecaec68ef1ff513b5ff8dcee2f29.txt
    │   │       ├── no_tumor_1288_jpg.rf.1716523f611f167a31ab8005753b985a.txt
    │   │       ├── no_tumor_1293_jpg.rf.3df5863ade1128b66f3e6b4f2b8f315e.txt
    │   │       ├── no_tumor_1297_jpg.rf.cebfcd4925e57647d14e1a8e00483656.txt
    │   │       ├── no_tumor_1309_jpg.rf.e92527e498844db62f674a25860351b4.txt
    │   │       ├── no_tumor_1325_jpg.rf.8874659d760783bb23c854f7112a65c6.txt
    │   │       ├── no_tumor_1326_jpg.rf.c8d7b79cda542c39fd6eb18013033052.txt
    │   │       ├── no_tumor_1330_jpg.rf.abd26941bb64c1899fa212b1f68d3f7f.txt
    │   │       ├── no_tumor_1346_jpg.rf.2e4aff010471355370cfa24a13c46516.txt
    │   │       ├── no_tumor_1351_jpg.rf.315a5f4d14e57f0c2ac119c53b2f93b0.txt
    │   │       ├── no_tumor_1356_jpg.rf.84735d5194d7ebaa17bc10f797435335.txt
    │   │       ├── no_tumor_1362_jpg.rf.229adcfa099b098f15ea935f83547cbc.txt
    │   │       ├── no_tumor_1375_jpg.rf.1cc6e4c7fced24a8e2b20c22761f5e10.txt
    │   │       ├── no_tumor_1376_jpg.rf.aff2f8a1edbd73173f348c43503fb461.txt
    │   │       ├── no_tumor_1377_jpg.rf.f1b71258d44b9c64d7d2ead7ca7db5d2.txt
    │   │       ├── no_tumor_1378_jpg.rf.c502c2a66eead244e8ca6d8131d565cd.txt
    │   │       ├── no_tumor_1381_jpg.rf.78b44b23406690cab08efe18368e6604.txt
    │   │       ├── no_tumor_1382_jpg.rf.e8ca4a15c262a636d8b9c58b2277075a.txt
    │   │       ├── no_tumor_1383_jpg.rf.83d5b9dea50a4b1c805b3dfeb523b25c.txt
    │   │       ├── no_tumor_1401_jpg.rf.000dba443c62539495f338739308ce48.txt
    │   │       ├── no_tumor_1402_jpg.rf.a1a01c9623bf180dc9b357d86dc4291f.txt
    │   │       ├── no_tumor_1417_jpg.rf.dc51cc6b0a36c02be55b38e6b9b32867.txt
    │   │       ├── no_tumor_1422_jpg.rf.f3bccc45e3099fbc62a02c4e79bc41e8.txt
    │   │       ├── no_tumor_1430_jpg.rf.f747c66cb2015e28af93695f56c4123c.txt
    │   │       ├── no_tumor_1432_jpg.rf.7adef0d175984efb42ffcd714fd528b8.txt
    │   │       ├── no_tumor_1435_jpg.rf.4d1db0e9166cc45301b63de06e43ab13.txt
    │   │       ├── no_tumor_1447_jpg.rf.196ca26893868d7b806177223b3913a9.txt
    │   │       ├── no_tumor_1448_jpg.rf.ced72720cad2eba824730db111df41c7.txt
    │   │       ├── no_tumor_1454_jpg.rf.916adb111444c20987e20200b9798eb7.txt
    │   │       ├── no_tumor_1457_jpg.rf.d5ebbc52838c88699a3a448e97dc8f3e.txt
    │   │       ├── no_tumor_1470_jpg.rf.c24b57bab59623fe01a2eb0857080204.txt
    │   │       ├── no_tumor_1473_jpg.rf.1888a952f485fcdd9a22c1a8d536464d.txt
    │   │       ├── no_tumor_1477_jpg.rf.e715b48cc76a3249b447c1d9bb61cefa.txt
    │   │       ├── no_tumor_1483_jpg.rf.3bebb7373057a2a28118160040aae5ab.txt
    │   │       ├── no_tumor_1493_jpg.rf.3d4d7b8079dcba963e761cc65fc20050.txt
    │   │       ├── no_tumor_1495_jpg.rf.239bffb0f85441405a6dca078ecfe55a.txt
    │   │       ├── no_tumor_1508_jpg.rf.e6ab7615bdeff09399d095c91a83cdd2.txt
    │   │       ├── no_tumor_1519_jpg.rf.5c6ac517d7c17eb648b83d17b19f5f66.txt
    │   │       ├── no_tumor_1532_jpg.rf.e4162f45c4274d225354927c0bf0bf41.txt
    │   │       ├── no_tumor_1546_jpg.rf.b78874b776c0b0dcccc9e8b6688a9342.txt
    │   │       ├── no_tumor_1556_jpg.rf.f1971f5978217035ed2789638b8338f1.txt
    │   │       ├── no_tumor_1559_jpg.rf.c32373b116b9af904e4a8bd66e7ea912.txt
    │   │       ├── no_tumor_1561_jpg.rf.505e7b45cfce48a858123334c9861cd9.txt
    │   │       ├── no_tumor_1565_jpg.rf.1dc6b0996167862af8d243634c16d21b.txt
    │   │       ├── no_tumor_1569_jpg.rf.bf40473a0dff4ca83c9b478a28bdede1.txt
    │   │       ├── no_tumor_156_jpg.rf.8e6c8f1ea47b6317d6dafd806860be0e.txt
    │   │       ├── no_tumor_1573_jpg.rf.b38a4104755fedebfa3b321f128d9ba2.txt
    │   │       ├── no_tumor_167_jpg.rf.c78f406de3d2740c6a21a1c26b4d4e73.txt
    │   │       ├── no_tumor_172_jpg.rf.2c442fd363c927d59e8aa8c598e69627.txt
    │   │       ├── no_tumor_176_jpg.rf.211bf548ffa63887aed9bdc6bedaf2f9.txt
    │   │       ├── no_tumor_17_jpg.rf.47d32fcac83806a34fc6a14db18749a2.txt
    │   │       ├── no_tumor_18_jpg.rf.1351a2b7f858de523903135931f21c2a.txt
    │   │       ├── no_tumor_196_jpg.rf.1e91801e190a5630099c4f0b1712d055.txt
    │   │       ├── no_tumor_208_jpg.rf.0854557a52f738f8a9a581ade19e84ab.txt
    │   │       ├── no_tumor_210_jpg.rf.fdb3c871e7fc31c638d1ae4622f6655f.txt
    │   │       ├── no_tumor_21_jpg.rf.68d32f400194ee91591f670f61f75e97.txt
    │   │       ├── no_tumor_222_jpg.rf.7cfb27d35a17497886dfe1ac98b66b0a.txt
    │   │       ├── no_tumor_227_jpg.rf.2630c3fc56c36439fdbd417a8a5f80cd.txt
    │   │       ├── no_tumor_239_jpg.rf.995bb1e62cc872f5b99f60049f4538ea.txt
    │   │       ├── no_tumor_274_jpg.rf.f693719103f198d85158ae961580ca22.txt
    │   │       ├── no_tumor_283_jpg.rf.c1f48ded5592810a26b5ac0c952bcdd0.txt
    │   │       ├── no_tumor_296_jpg.rf.37d6dc5e9f6875ae8972c9433efdc70a.txt
    │   │       ├── no_tumor_303_jpg.rf.f1d183452b4a78c0253cb066fccb1a73.txt
    │   │       ├── no_tumor_306_jpg.rf.a0b6f5a007474d22a0c7825b30c12c6c.txt
    │   │       ├── no_tumor_30_jpg.rf.df215710a36490ef52b7ddfaf254c1c3.txt
    │   │       ├── no_tumor_310_jpg.rf.a8f3513a653d2a692f87c5d8258e3f1c.txt
    │   │       ├── no_tumor_327_jpg.rf.e1977eb52d77e222443eeb2210f3c866.txt
    │   │       ├── no_tumor_333_jpg.rf.fccfc15a10aea59ee0201b416b5fbc54.txt
    │   │       ├── no_tumor_349_jpg.rf.7b928feaaaf67a568893890d8a32e000.txt
    │   │       ├── no_tumor_374_jpg.rf.45c3f5586a1f96c321c3e6a6be475732.txt
    │   │       ├── no_tumor_380_jpg.rf.fe6399bad33fea5225c6bd50d4c81c70.txt
    │   │       ├── no_tumor_389_jpg.rf.eb17318825b175228ac172bb0919a71b.txt
    │   │       ├── no_tumor_395_jpg.rf.8611304494e5f48b2b5bdee1681f17d0.txt
    │   │       ├── no_tumor_400_jpg.rf.9ef016ab9dfe63e92695bcb66c66c089.txt
    │   │       ├── no_tumor_405_jpg.rf.615ee173a7516905e28a71c8075814a2.txt
    │   │       ├── no_tumor_406_jpg.rf.7f1557f0e11ab67ae8859c327f09cf52.txt
    │   │       ├── no_tumor_411_jpg.rf.8fb8cd910f986e91aca953f4badb7285.txt
    │   │       ├── no_tumor_416_jpg.rf.119864a2eece4b1ae1e956cbff256020.txt
    │   │       ├── no_tumor_420_jpg.rf.9d4673b08e0deb85104ac03184eda772.txt
    │   │       ├── no_tumor_430_jpg.rf.af0e9f050f5cb81fc5df3cd867d32523.txt
    │   │       ├── no_tumor_457_jpg.rf.96965937cd4c3ab8c2fedc376e998dff.txt
    │   │       ├── no_tumor_468_jpg.rf.87ab342161dd8d47ec77853a0f40c082.txt
    │   │       ├── no_tumor_473_jpg.rf.ab7cc6c12b14c280edd149b89eb4c0a4.txt
    │   │       ├── no_tumor_481_jpg.rf.0f3734afa1d75ea772750755d55d62af.txt
    │   │       ├── no_tumor_485_jpg.rf.5ae3d38860013bc53cd9c2cdfcade36c.txt
    │   │       ├── no_tumor_486_jpg.rf.38c0a3771298af140f9f9478009f49e9.txt
    │   │       ├── no_tumor_533_jpg.rf.e7545dd4398fa1863e69fc810d45f00d.txt
    │   │       ├── no_tumor_538_jpg.rf.680e5f1c88286ecbd7433465e9b95b6d.txt
    │   │       ├── no_tumor_556_jpg.rf.cf153f32249e1cbd3a50e594af847272.txt
    │   │       ├── no_tumor_574_jpg.rf.ee8c2c9fb8a2e5f61947a418397ea6f3.txt
    │   │       ├── no_tumor_582_jpg.rf.6272f3e2fef0ce02f3da23ae0b9eee8a.txt
    │   │       ├── no_tumor_586_jpg.rf.50372a07323a4714564e4b908b5a1116.txt
    │   │       ├── no_tumor_587_jpg.rf.bb4bece0c4ba89ffadc4a9da5d780b7d.txt
    │   │       ├── no_tumor_58_jpg.rf.65cbd9ab986d64b73ec16aa4de1e42b3.txt
    │   │       ├── no_tumor_591_jpg.rf.3a2c2b34baf0239f388124c41cfb62f6.txt
    │   │       ├── no_tumor_593_jpg.rf.4dd8f671d519a783966a473992fd0ff7.txt
    │   │       ├── no_tumor_601_jpg.rf.daafa5bad1f585f666ca475a583d1649.txt
    │   │       ├── no_tumor_605_jpg.rf.e722600b49a9c1465deaaf5c11599292.txt
    │   │       ├── no_tumor_610_jpg.rf.1ad423b5823a71bf739a909d7065f5cd.txt
    │   │       ├── no_tumor_61_jpg.rf.2b8a81a6ad68256a0cc0204cbc55df6a.txt
    │   │       ├── no_tumor_625_jpg.rf.aed2835c83bf66f675373c82b57c5c31.txt
    │   │       ├── no_tumor_634_jpg.rf.e499d2363eec7206db37a25261614a96.txt
    │   │       ├── no_tumor_638_jpg.rf.db54847287e9389b62df0d9f36400b01.txt
    │   │       ├── no_tumor_640_jpg.rf.778d020be48fc5d6f89b1fdaa06531a4.txt
    │   │       ├── no_tumor_689_jpg.rf.b93d4818eca1c52a004e876d2ff69881.txt
    │   │       ├── no_tumor_698_jpg.rf.3c0efb1fb3525c0ef805bb505130b347.txt
    │   │       ├── no_tumor_712_jpg.rf.7c9beb139eaeef812a1e03655d4ebe6a.txt
    │   │       ├── no_tumor_716_jpg.rf.68ab707e5d1310f0a3edb3cb5c1b8bb0.txt
    │   │       ├── no_tumor_720_jpg.rf.0ae4144b6b9d6f92f062e0572363fbb6.txt
    │   │       ├── no_tumor_731_jpg.rf.9edfa6052c902c46fd00b63f823cfa52.txt
    │   │       ├── no_tumor_732_jpg.rf.3c474029a8875f7f6d2e0f57a1cd7c55.txt
    │   │       ├── no_tumor_738_jpg.rf.0b4c4930b7a5b70b20147eaa1f5a7a11.txt
    │   │       ├── no_tumor_749_jpg.rf.d4602efed782d5dabb2361c2a4de72d6.txt
    │   │       ├── no_tumor_74_jpg.rf.9a21362dc700814773b0d227d15ac891.txt
    │   │       ├── no_tumor_762_jpg.rf.8c71bc2d60f76136bec76600a41d3877.txt
    │   │       ├── no_tumor_765_jpg.rf.350e4653aca3276394180480ff9af8e1.txt
    │   │       ├── no_tumor_776_jpg.rf.d61b6a7cbab2784da79110fcf613ff50.txt
    │   │       ├── no_tumor_779_jpg.rf.43be10e011c63f87810a9c2bef9a85ec.txt
    │   │       ├── no_tumor_785_jpg.rf.d6e18a06d2588adc74b2b0d25ba1a284.txt
    │   │       ├── no_tumor_78_jpg.rf.fd979dc240879db328f9abd46febcda1.txt
    │   │       ├── no_tumor_802_jpg.rf.07031c4a1ce5d9c87a5b5ebd754a9373.txt
    │   │       ├── no_tumor_805_jpg.rf.07d98ea520cc63e0d83d9fb307511dc0.txt
    │   │       ├── no_tumor_812_jpg.rf.895d53fe327f2d4123f787e9282daeb0.txt
    │   │       ├── no_tumor_814_jpg.rf.c75d6c6bb5fa8e38659cedf59e3397b3.txt
    │   │       ├── no_tumor_828_jpg.rf.0bd5c0221ab1fcd8f6bcd598e21b57fe.txt
    │   │       ├── no_tumor_833_jpg.rf.b51277ee33ce606516cd652c21a0dc1c.txt
    │   │       ├── no_tumor_83_jpg.rf.bb18a612f96e88b50dde87c657ec7e3c.txt
    │   │       ├── no_tumor_867_jpg.rf.3dbcb4d0ee6596fd6d8ab68c5415a5ac.txt
    │   │       ├── no_tumor_871_jpg.rf.87ed23d46cc3aac75b4492934b5fcc70.txt
    │   │       ├── no_tumor_877_jpg.rf.6851911587ed126a7e017d207218587b.txt
    │   │       ├── no_tumor_880_jpg.rf.c42192f2b2f546c0b1da245173dd798a.txt
    │   │       ├── no_tumor_881_jpg.rf.e4296a6bf08b8dae6ca3436cdbff68c4.txt
    │   │       ├── no_tumor_904_jpg.rf.73f0ee7d9426dddcb49489d5e4676928.txt
    │   │       ├── no_tumor_914_jpg.rf.6714544aee2bde5213fd2c366dff62aa.txt
    │   │       ├── no_tumor_915_jpg.rf.9b3b8ca77f41c0d77cd67d05d4b39bf1.txt
    │   │       ├── no_tumor_920_jpg.rf.2b10abebde1cf6bca35139c4fe34478c.txt
    │   │       ├── no_tumor_932_jpg.rf.a6dc07dc604c03e84f7224782a6cdce9.txt
    │   │       ├── no_tumor_934_jpg.rf.d1cd0942b91c29942b34b7693c96adb8.txt
    │   │       ├── no_tumor_960_jpg.rf.f968706e3f72df5eb006f51fcdec01b0.txt
    │   │       ├── no_tumor_973_jpg.rf.090d284c0592a1cfe32f4348c26d126c.txt
    │   │       ├── no_tumor_97_jpg.rf.932de25b4dfeb8a7b4bab146c57b1a3b.txt
    │   │       ├── no_tumor_982_jpg.rf.63ad40d046a68124bca367b4e8d111d3.txt
    │   │       ├── no_tumor_992_jpg.rf.9c52cd6b6948f0c7d4731a65ec2350b5.txt
    │   │       ├── no_tumor_9_jpg.rf.f12a0c01706e830ae6441bb9eab92796.txt
    │   │       ├── pituitary_1010_jpg.rf.4b32ede821cc21169ee7ee87c7475f8b.txt
    │   │       ├── pituitary_1024_jpg.rf.3109ab187ccbc49368a10fad74be2637.txt
    │   │       ├── pituitary_1027_jpg.rf.61b9436cbddca07208253f7fb77dea3c.txt
    │   │       ├── pituitary_1030_jpg.rf.f6c51f77adf6afb8cc8fd9e6ec491e09.txt
    │   │       ├── pituitary_1033_jpg.rf.f3674beb04a997d22e70560719dad517.txt
    │   │       ├── pituitary_1038_jpg.rf.d8fa0cf57747c0fa5270fce9ff9003cd.txt
    │   │       ├── pituitary_1039_jpg.rf.da3dee214dabb24e4dd95db2b53ec015.txt
    │   │       ├── pituitary_103_jpg.rf.bff7d4ea3ad4757eb75a3204a97b2285.txt
    │   │       ├── pituitary_1042_jpg.rf.1c6a9f22d4656dc9f913e1321233be2b.txt
    │   │       ├── pituitary_1061_jpg.rf.73c2839491aecd2d6740e17c1e01a08a.txt
    │   │       ├── pituitary_1064_jpg.rf.c806771afa32d21b6cc68e8707be8385.txt
    │   │       ├── pituitary_1066_jpg.rf.40ddf908fa6d7c77f89bb79611c83c1c.txt
    │   │       ├── pituitary_1068_jpg.rf.edaf0b6b7d3b2985eda979ed6e841982.txt
    │   │       ├── pituitary_1078_jpg.rf.2cd1f412ca11bb69c6e684468ef5b211.txt
    │   │       ├── pituitary_1085_jpg.rf.ad625462177f73ddc314dd946a6292e2.txt
    │   │       ├── pituitary_1087_jpg.rf.0b03feac0d2d2fc41b4f53e85d277900.txt
    │   │       ├── pituitary_1089_jpg.rf.438fabf0bedd684773e3815fa8a7a8e3.txt
    │   │       ├── pituitary_1101_jpg.rf.81ca54811968652994121d9bcb03111a.txt
    │   │       ├── pituitary_1111_jpg.rf.1e35a2f8c97d6330c6b2b4c6e7ca38d7.txt
    │   │       ├── pituitary_1124_jpg.rf.bd04ab69a51815d24906c9e24006d994.txt
    │   │       ├── pituitary_113_jpg.rf.9a22b4ebeb1132033df1f5184951f6eb.txt
    │   │       ├── pituitary_1148_jpg.rf.456fb9d350403fbe6095508093ddce49.txt
    │   │       ├── pituitary_114_jpg.rf.dab702f45daa231783befdf032cae55c.txt
    │   │       ├── pituitary_1177_jpg.rf.35da09d3f616eeda96cb4e7609a423db.txt
    │   │       ├── pituitary_1181_jpg.rf.f46ac8ce835c7328a64bf0a3a7771e79.txt
    │   │       ├── pituitary_1188_jpg.rf.d22566a2b264e6995db3a783b303c8b9.txt
    │   │       ├── pituitary_11_jpg.rf.bde1d9fcca94f9c92d0378690a700e61.txt
    │   │       ├── pituitary_1206_jpg.rf.3d813be76058ce3c9d5727a3d008d368.txt
    │   │       ├── pituitary_1207_jpg.rf.339dcbf52551b285173535c6c18a040c.txt
    │   │       ├── pituitary_1236_jpg.rf.d041da130c5eede6ef69dc356eb82620.txt
    │   │       ├── pituitary_1238_jpg.rf.75ded478842a9b689f4c0ad285632be3.txt
    │   │       ├── pituitary_1242_jpg.rf.78cb91ce4d04f66626021099d15dc5bd.txt
    │   │       ├── pituitary_124_jpg.rf.02166d4ed6c79fa7861917ce569a6ebd.txt
    │   │       ├── pituitary_1259_jpg.rf.e03d0d37b7d24d62f1085443a377b89d.txt
    │   │       ├── pituitary_1261_jpg.rf.492393a04066b17cdb6b60d080852e1d.txt
    │   │       ├── pituitary_1279_jpg.rf.5515a4c82ff4beb3a384ae1ca177d8d1.txt
    │   │       ├── pituitary_1282_jpg.rf.9ee76eab16909943dc2d105ea0669e10.txt
    │   │       ├── pituitary_1299_jpg.rf.e0a6171844c4f1d06a5af3a6037b3ab6.txt
    │   │       ├── pituitary_1309_jpg.rf.6bae60f8182afc580ca187c230fd12b7.txt
    │   │       ├── pituitary_1317_jpg.rf.8c09c8a2c50efed5cd9390131e44288f.txt
    │   │       ├── pituitary_1342_jpg.rf.b792feb6aad2cba6c1326b4d6077fef9.txt
    │   │       ├── pituitary_1343_jpg.rf.abb4d43d5f4e56c6e6fafc98adcda5fa.txt
    │   │       ├── pituitary_1346_jpg.rf.bd82a62829e4af57da80bc166d55a33d.txt
    │   │       ├── pituitary_1353_jpg.rf.41935af18f13bf8b58005596af4179fc.txt
    │   │       ├── pituitary_137_jpg.rf.9039b0515d9213947be053137f97cf98.txt
    │   │       ├── pituitary_1382_jpg.rf.2429a0c6bc4404c7c7647eb9144905db.txt
    │   │       ├── pituitary_1383_jpg.rf.c9d98fe5e7fea02b05ab0de87cc2f020.txt
    │   │       ├── pituitary_1404_jpg.rf.dbe8debedafd2bfc0fe4803699dc829e.txt
    │   │       ├── pituitary_1405_jpg.rf.f8c8c2496cdbaf8a83a97e5f4bcc8f1c.txt
    │   │       ├── pituitary_1407_jpg.rf.09b2eaf7e619a9c235e3048c3bd3f0a6.txt
    │   │       ├── pituitary_1409_jpg.rf.db632f15e69b86603d45b0a408b638ae.txt
    │   │       ├── pituitary_1415_jpg.rf.3dcfacd6d61c498a37ae104a7b68484a.txt
    │   │       ├── pituitary_1421_jpg.rf.5b382942d4944afae4a203eb2ca6eeb8.txt
    │   │       ├── pituitary_1427_jpg.rf.5595b63eea9fb4a00d5581ae62ab47ec.txt
    │   │       ├── pituitary_1436_jpg.rf.6fbbb593c8e651b7ba90ecc98d688f09.txt
    │   │       ├── pituitary_1439_jpg.rf.52e40f4eac3531a59fe34d0037993d9e.txt
    │   │       ├── pituitary_1451_jpg.rf.abba75cfa7ccfb3aeac67735105d4d6c.txt
    │   │       ├── pituitary_146_jpg.rf.fb709ef6cc597e964dddb0ca4824f553.txt
    │   │       ├── pituitary_147_jpg.rf.d7addc2328755de0d735cd46731b550f.txt
    │   │       ├── pituitary_14_jpg.rf.0ed43606cb64d0d8fb914beb9b3d71b1.txt
    │   │       ├── pituitary_152_jpg.rf.fee1aa85a22eb1514164cafefab9ef37.txt
    │   │       ├── pituitary_157_jpg.rf.fab3a62e15ce58bd6d8dcb1c462f48e7.txt
    │   │       ├── pituitary_158_jpg.rf.77c6fe774b3e1cddd6476bc3bedd447c.txt
    │   │       ├── pituitary_16_jpg.rf.b84c67869927f85ff477ae7db6db9a5c.txt
    │   │       ├── pituitary_170_jpg.rf.1b460d77ac26e2c707ff8997576dd03b.txt
    │   │       ├── pituitary_19_jpg.rf.13cadca08978ec32b69bed7772dd3ea1.txt
    │   │       ├── pituitary_202_jpg.rf.b72bd3452c676440444e734d19732718.txt
    │   │       ├── pituitary_20_jpg.rf.5a0ebcd704a580a4392104b5d6d706f8.txt
    │   │       ├── pituitary_21_jpg.rf.bf154eb052afca2a3503d6ac4e3809df.txt
    │   │       ├── pituitary_243_jpg.rf.2207eb779fccb72755c361bc2ad5b425.txt
    │   │       ├── pituitary_244_jpg.rf.58478c0560e03a47e4f680a098b10d0a.txt
    │   │       ├── pituitary_247_jpg.rf.3747d6cdc2b67c4ee7e8b5f82f23e8be.txt
    │   │       ├── pituitary_249_jpg.rf.3953374b4c05261b5e7f1ea7460238ff.txt
    │   │       ├── pituitary_254_jpg.rf.097b1debae13249fafc0b6f320034392.txt
    │   │       ├── pituitary_260_jpg.rf.6106032205be1c58d46f2b42244de930.txt
    │   │       ├── pituitary_269_jpg.rf.9035e3b437756afc4755b01d129965b0.txt
    │   │       ├── pituitary_277_jpg.rf.7575731b5aea84cd117d7395cddbe672.txt
    │   │       ├── pituitary_280_jpg.rf.27a5e63dca0569f36558ddd3f4198e77.txt
    │   │       ├── pituitary_295_jpg.rf.33f023252628dc40b0e4f044df06ba0b.txt
    │   │       ├── pituitary_2_jpg.rf.b59a4592aebe10effa30a561108513dc.txt
    │   │       ├── pituitary_314_jpg.rf.f9614bef9231de2a434382d7f7f5361b.txt
    │   │       ├── pituitary_339_jpg.rf.5bfb40b325964b3de4951e93b9097a44.txt
    │   │       ├── pituitary_347_jpg.rf.5b5a6738b4564e26e2ac4a12a83fe71f.txt
    │   │       ├── pituitary_349_jpg.rf.f6d0644ef3211c291e4b7ccdb217c48b.txt
    │   │       ├── pituitary_352_jpg.rf.3a2546e2cf83cbda2540be0c96ec5ab1.txt
    │   │       ├── pituitary_353_jpg.rf.d4c4f71325bfc241e61864a4848251a5.txt
    │   │       ├── pituitary_362_jpg.rf.cbd51c6049ef39b3dafdcd32c7122718.txt
    │   │       ├── pituitary_366_jpg.rf.17aeda00d23ceea54c56aad7da13f09b.txt
    │   │       ├── pituitary_367_jpg.rf.6def2e69ca7ff611b14c49c62d099a33.txt
    │   │       ├── pituitary_383_jpg.rf.8f63a70e496e5263199823752e7bc500.txt
    │   │       ├── pituitary_395_jpg.rf.6ef2fc99bbf408189d3afe1c223c6896.txt
    │   │       ├── pituitary_425_jpg.rf.657eea891f15b8c4358c3efe83a83aad.txt
    │   │       ├── pituitary_429_jpg.rf.6820785128fc5fb59d697c85a276689e.txt
    │   │       ├── pituitary_434_jpg.rf.82eb679432f78789b1e9a4e1f979cb04.txt
    │   │       ├── pituitary_442_jpg.rf.08e48ca357ff70abbfa5bc1ce17f45a8.txt
    │   │       ├── pituitary_444_jpg.rf.616b3837eb98648e1c60e7920eae79bb.txt
    │   │       ├── pituitary_449_jpg.rf.6d75373e9387e79e037bdac0438c0e6f.txt
    │   │       ├── pituitary_44_jpg.rf.c48004d2e5932f2334c827524d658c8b.txt
    │   │       ├── pituitary_465_jpg.rf.f3145501ee162dfcbecbafa0153ebefc.txt
    │   │       ├── pituitary_473_jpg.rf.e1948b14f933edfeade450f357db08f3.txt
    │   │       ├── pituitary_482_jpg.rf.dd6b990733496ceda8f4fad4a9d438e7.txt
    │   │       ├── pituitary_516_jpg.rf.3145996fbbb24667aac495d31785f94f.txt
    │   │       ├── pituitary_519_jpg.rf.f98a7a057186388f5dabbce0af86eac0.txt
    │   │       ├── pituitary_51_jpg.rf.c86c7c9cb6f00d5d6a5ebf6374ff55a0.txt
    │   │       ├── pituitary_55_jpg.rf.4973b5b39b5831b2fae3112a0764d24c.txt
    │   │       ├── pituitary_566_jpg.rf.36dc9507f01dda158361fec1e1656203.txt
    │   │       ├── pituitary_571_jpg.rf.9932cf26092b727e163954279e53f3f3.txt
    │   │       ├── pituitary_575_jpg.rf.797a6694b0f849b807d6bc0761462b06.txt
    │   │       ├── pituitary_588_jpg.rf.f2ced73df65f8bb544a762fb33e72f0e.txt
    │   │       ├── pituitary_590_jpg.rf.e84608167ffa04939c2bb714fc8380a5.txt
    │   │       ├── pituitary_596_jpg.rf.554a600b1f88405d132b160f1229ad36.txt
    │   │       ├── pituitary_612_jpg.rf.68bf84d673ece7bddabaef4ff4892665.txt
    │   │       ├── pituitary_613_jpg.rf.fb80ec77fc161e85db9485b9ca93520d.txt
    │   │       ├── pituitary_632_jpg.rf.8fdf813e1806f69c947425bbbf6239dd.txt
    │   │       ├── pituitary_633_jpg.rf.996b1eb1f3d572ca7f8638174a69f144.txt
    │   │       ├── pituitary_635_jpg.rf.9aa64b0eb54322fe8965104ada04de79.txt
    │   │       ├── pituitary_636_jpg.rf.4adb9120661b9d4e1728ebe83b4cabc4.txt
    │   │       ├── pituitary_653_jpg.rf.2f24641dd8e8005ef537e44e45e62962.txt
    │   │       ├── pituitary_687_jpg.rf.f0006d3aae44dc3214e24824c1266c6b.txt
    │   │       ├── pituitary_694_jpg.rf.19db15b9effa9045595c060a8a99c38f.txt
    │   │       ├── pituitary_696_jpg.rf.90761be4dcae375307c1effc13d35364.txt
    │   │       ├── pituitary_6_jpg.rf.2ca208e02910662c1dae7d012c5ae6c7.txt
    │   │       ├── pituitary_709_jpg.rf.d810bc61ee13a6dac8cc402cfdf485b5.txt
    │   │       ├── pituitary_736_jpg.rf.a6597967353c35a003c6b083d31835e0.txt
    │   │       ├── pituitary_745_jpg.rf.db50f0c0c6d89535fd892b476f3bdc0e.txt
    │   │       ├── pituitary_747_jpg.rf.1cd94a1e17168f351aefbd2e1aecaaa4.txt
    │   │       ├── pituitary_766_jpg.rf.bd99f9b94c920fe79998c62f45f9e47d.txt
    │   │       ├── pituitary_768_jpg.rf.e8a430da7c5b0950e7df9462e5e4de26.txt
    │   │       ├── pituitary_784_jpg.rf.35fbd8d5e826518335d6983657dafb52.txt
    │   │       ├── pituitary_796_jpg.rf.9c5dc3279b211487c125bfb7cf39a3b8.txt
    │   │       ├── pituitary_811_jpg.rf.1615ed84247b9d492b270de36e664e8e.txt
    │   │       ├── pituitary_812_jpg.rf.8049b42fd2d7555107945f85d74f0cb0.txt
    │   │       ├── pituitary_817_jpg.rf.4bc6eec52131d9d6e2dac873123dd565.txt
    │   │       ├── pituitary_831_jpg.rf.591ba244eb69a0d6bd26de903a4df1bf.txt
    │   │       ├── pituitary_838_jpg.rf.82b13975e57d46c90838e2b4e6aa77ef.txt
    │   │       ├── pituitary_860_jpg.rf.91c08cf0b866472922fc62d1bf8e086b.txt
    │   │       ├── pituitary_862_jpg.rf.a469356b04ff392e126c0d56d8eb52e0.txt
    │   │       ├── pituitary_883_jpg.rf.41b151d1be030e3197206cc341a8f3b8.txt
    │   │       ├── pituitary_884_jpg.rf.a62e2da57ac8687dad3815d5588d1f2e.txt
    │   │       ├── pituitary_890_jpg.rf.400ed548e401014b561dd674f5b793d7.txt
    │   │       ├── pituitary_921_jpg.rf.4f9bc7ce14afa06a9f462f4b1e052962.txt
    │   │       ├── pituitary_930_jpg.rf.bab1001bd2e968752716a86bae781090.txt
    │   │       ├── pituitary_932_jpg.rf.aa24e9a8430e9d1c448c74ae8e37e70d.txt
    │   │       ├── pituitary_940_jpg.rf.54a9faf38d85dc767cf4a9ce0af0d5f0.txt
    │   │       ├── pituitary_961_jpg.rf.1563674e159c7cb834396022137202dc.txt
    │   │       ├── pituitary_976_jpg.rf.c2692a31bc999aeaeec2fe9e844e173a.txt
    │   │       └── pituitary_997_jpg.rf.e360b7003605a483342546daed1d7f8e.txt
    │   └── valid/
    │       ├── labels.cache
    │       ├── images/
    │       └── labels/
    │           ├── glioma_1022_jpg.rf.ab6956aa8c8a25f994539c5cf6227064.txt
    │           ├── glioma_104_jpg.rf.f5deabc016a8b3a913888b09baebacee.txt
    │           ├── glioma_1088_jpg.rf.5542c8b3dc2add56cd7303d7007e3ae8.txt
    │           ├── glioma_1109_jpg.rf.710d66962bf0db65050c34e750be6e7a.txt
    │           ├── glioma_1164_jpg.rf.4f2cfa1dc0e93548eeef3f9c30e3b7ee.txt
    │           ├── glioma_1226_jpg.rf.b459e0d24ecd906c121e9be1b88907c4.txt
    │           ├── glioma_1238_jpg.rf.e7fec135cbd5bafba3674a18b0b85818.txt
    │           ├── glioma_1254_jpg.rf.809e3b791a236a04a1445e2c5f7d979e.txt
    │           ├── glioma_1255_jpg.rf.ede8d4e550bf157ed2c9faf53dbeef4c.txt
    │           ├── glioma_1265_jpg.rf.0ceada97135f316a54f843aed7d1703e.txt
    │           ├── glioma_1275_jpg.rf.b57270922a310c95831b5373c35f7e1f.txt
    │           ├── glioma_12_jpg.rf.f146c6663e2614eba9da724a1c495acd.txt
    │           ├── glioma_1305_jpg.rf.9e8e8139db1dc7b685a00a340c0dcafd.txt
    │           ├── glioma_154_jpg.rf.d6b8dbf3c9061876f1ee6dbbe6113664.txt
    │           ├── glioma_167_jpg.rf.da25e4d04a053b942c6c5211a14397b3.txt
    │           ├── glioma_201_jpg.rf.a847f9af4f209d6323a0c4e4f07d63f1.txt
    │           ├── glioma_217_jpg.rf.8f0465bb2b7c3b57b0df37ff73a0259c.txt
    │           ├── glioma_334_jpg.rf.08fe123f3cb4647ece300fc6aa648214.txt
    │           ├── glioma_383_jpg.rf.c56f80ec1efa67d854350344b04f391d.txt
    │           ├── glioma_40_jpg.rf.a578edbb1cfe59a55b8377cdf4e46f16.txt
    │           ├── glioma_428_jpg.rf.2d27302e3d74586debc5ab63e605671f.txt
    │           ├── glioma_477_jpg.rf.fac2810f0b518f328de3369818336daf.txt
    │           ├── glioma_545_jpg.rf.1b2c558abef09235677541b8709a12b2.txt
    │           ├── glioma_620_jpg.rf.aad32c4d1721799b3ad1149352cd3af5.txt
    │           ├── glioma_639_jpg.rf.b9c17f063e393ff2952e662a4af07ee2.txt
    │           ├── glioma_73_jpg.rf.c59c51554d141873ad71c7482447e854.txt
    │           ├── glioma_76_jpg.rf.525d4172ac7731bdb68a73504014ffcc.txt
    │           ├── glioma_937_jpg.rf.ae7c14cefc5df10cfd7a4a4c34ddfdf4.txt
    │           ├── glioma_948_jpg.rf.f28cc5f42f837b7e005ffb2c740b8160.txt
    │           ├── glioma_964_jpg.rf.fc8440cfc0c94c2d452edebb7f71f0c5.txt
    │           ├── meningioma_1018_jpg.rf.31a71a9999537db1e0651387f1c2d102.txt
    │           ├── meningioma_1025_jpg.rf.642e70889054d7e7f3048e46948630a1.txt
    │           ├── meningioma_1044_jpg.rf.312ef7316cddb11bd721766f1712eced.txt
    │           ├── meningioma_108_jpg.rf.4b7febff023e20c52b0bda99f20322aa.txt
    │           ├── meningioma_1123_jpg.rf.51f067df444ea84a088907c87805bdbe.txt
    │           ├── meningioma_1125_jpg.rf.a8bfa701cc1fe4ce49a9e261ca79d4dd.txt
    │           ├── meningioma_115_jpg.rf.98156ea3bf9d25bbe5bc18bc6cc483b8.txt
    │           ├── meningioma_1173_jpg.rf.b612f4d2f294a442496a0c61719c2c4b.txt
    │           ├── meningioma_1184_jpg.rf.788f4bed9003550b08637f1f22bff24c.txt
    │           ├── meningioma_1189_jpg.rf.b46edce8b68e411674c47b753789fff6.txt
    │           ├── meningioma_1196_jpg.rf.1ea0c6622f1af2932f65b9ae6d07b1f4.txt
    │           ├── meningioma_1198_jpg.rf.aa562bb4304e3a28543e0715ddf77190.txt
    │           ├── meningioma_1202_jpg.rf.8006550acebd969ce2579747633630cc.txt
    │           ├── meningioma_1205_jpg.rf.7a3551dd2ce9a35906dc1ae386c37e2d.txt
    │           ├── meningioma_1208_jpg.rf.48b72738c07709590a0bde122c672a67.txt
    │           ├── meningioma_1211_jpg.rf.7800f5eeb077d1398a44ce7edfbdec8c.txt
    │           ├── meningioma_1215_jpg.rf.b507c5e7df19598a05db96b13cc71c75.txt
    │           ├── meningioma_1223_jpg.rf.5ca4caf12a46da940adeba0c67a7f2b8.txt
    │           ├── meningioma_122_jpg.rf.b8f33c11af9abf5b2d3a732f436b74e9.txt
    │           ├── meningioma_1232_jpg.rf.461290cf338a9e9c13e2e11757d97c9a.txt
    │           ├── meningioma_1241_jpg.rf.addad1eea8a59b7111e1a2c2ab052b97.txt
    │           ├── meningioma_1244_jpg.rf.d7d2600434c1faab0d4d353498543e2d.txt
    │           ├── meningioma_1245_jpg.rf.a4b95728201624d4cf56e70a12fa288f.txt
    │           ├── meningioma_1276_jpg.rf.7bae1a07294c81b208e6eb56c40ee365.txt
    │           ├── meningioma_1285_jpg.rf.16719575be6242b178b7258a6f12e940.txt
    │           ├── meningioma_1287_jpg.rf.05edb14dc7377ce1acf214e91375d452.txt
    │           ├── meningioma_1299_jpg.rf.215ba961935cafb3806427e0c46c4b39.txt
    │           ├── meningioma_1306_jpg.rf.9bb2c50d97af1fa023cb90661ffbad81.txt
    │           ├── meningioma_1315_jpg.rf.a6dae6a57801e04a0b060c32542d2276.txt
    │           ├── meningioma_1317_jpg.rf.9afa257fe8f51e66ba43459099b5c27e.txt
    │           ├── meningioma_1324_jpg.rf.4d5bff39778ccec03a572bb2f000e6b3.txt
    │           ├── meningioma_132_jpg.rf.2133ab6c2698ba0e76b75643b73b3fa6.txt
    │           ├── meningioma_142_jpg.rf.bc16b000d35742306f2a1788dbb00a1d.txt
    │           ├── meningioma_145_jpg.rf.af82b78e0d994f43a916bc2bf6527cf5.txt
    │           ├── meningioma_146_jpg.rf.29c24314852b50f3f04a4de7735c8261.txt
    │           ├── meningioma_14_jpg.rf.1d5c9e2741a6af8a523f877d2e1d0050.txt
    │           ├── meningioma_160_jpg.rf.04837c098df7a5f0184c80af849b8401.txt
    │           ├── meningioma_179_jpg.rf.2f2fd849ee977f3d5c781dddb57725e8.txt
    │           ├── meningioma_181_jpg.rf.a07e8ed8c0b3ee827c1e67d3fd60241b.txt
    │           ├── meningioma_184_jpg.rf.a0cf04543cb6542980ef171f2a9a2fc7.txt
    │           ├── meningioma_199_jpg.rf.ab708827b573d2f27462da723fc4209e.txt
    │           ├── meningioma_200_jpg.rf.1c8b789dd67a69f1e6fdd3c44f5fdb56.txt
    │           ├── meningioma_209_jpg.rf.0fa4494baf364d68baff0ad0730987a9.txt
    │           ├── meningioma_211_jpg.rf.6131e84721c8c3c95f46a3c158e27b1d.txt
    │           ├── meningioma_213_jpg.rf.24e73bf7cbf7beef2c82aeba6d81623c.txt
    │           ├── meningioma_220_jpg.rf.6b95313f6e1d3412f2809acf3d1f87ce.txt
    │           ├── meningioma_228_jpg.rf.2728d4bdea8f0dd8d364538ad2f939b0.txt
    │           ├── meningioma_230_jpg.rf.5d11a5fb2c30e4f584732d516838c23c.txt
    │           ├── meningioma_237_jpg.rf.446ceb32c2ae6efa17f77c23969dd9ee.txt
    │           ├── meningioma_241_jpg.rf.1a532ae953b9a685d99d20cb4e59f433.txt
    │           ├── meningioma_247_jpg.rf.4e654fe42faa65584d0d20217100cbe4.txt
    │           ├── meningioma_252_jpg.rf.77f7e2a406897946dc65db7a52515195.txt
    │           ├── meningioma_253_jpg.rf.e3e4cd09323e06fa5183418d371c3b1d.txt
    │           ├── meningioma_257_jpg.rf.a6b03ddbfee225035425cf2886ab5ee3.txt
    │           ├── meningioma_274_jpg.rf.e9a45742a67a9992a8fee774e8feb986.txt
    │           ├── meningioma_277_jpg.rf.80a7f9edd64bd084710a1abf65be1fe5.txt
    │           ├── meningioma_284_jpg.rf.a29a234784d4b2a3011332b28523ccda.txt
    │           ├── meningioma_311_jpg.rf.59ad21f3f40ec08bf507e0c81b11bb27.txt
    │           ├── meningioma_326_jpg.rf.06ff6187aa7b2837ab2ca0f1f6409133.txt
    │           ├── meningioma_327_jpg.rf.ae5ab1c26ed22d282ec65d161fcd34ba.txt
    │           ├── meningioma_332_jpg.rf.01427d915fbaf63631281218b34e67e0.txt
    │           ├── meningioma_336_jpg.rf.03306e22d13f8033ad873bfcd32f019f.txt
    │           ├── meningioma_337_jpg.rf.38c432304d491168a629d5114392583f.txt
    │           ├── meningioma_338_jpg.rf.d28d419f8acd2d514b7863bad0cbdaf7.txt
    │           ├── meningioma_340_jpg.rf.85d39b3cfc1e67bc21ade37d06bed6fd.txt
    │           ├── meningioma_350_jpg.rf.0e927d84952deb34014816419a8ae64f.txt
    │           ├── meningioma_351_jpg.rf.b26dc300f6a32c38d64ebedd50e55c72.txt
    │           ├── meningioma_356_jpg.rf.f31001a90c81278e3f9935663f2d9cdb.txt
    │           ├── meningioma_370_jpg.rf.dbec04b254ffcba58a97515c4fb72485.txt
    │           ├── meningioma_400_jpg.rf.2be92a05ad5b03f94b6ed8b649d80970.txt
    │           ├── meningioma_407_jpg.rf.563353fae3c19585fa4879834974c35f.txt
    │           ├── meningioma_408_jpg.rf.ac070051268f08ebd57d8502cdcc62d6.txt
    │           ├── meningioma_432_jpg.rf.142ba8cd81a1ad3e876b9d2df40ac99b.txt
    │           ├── meningioma_438_jpg.rf.e8882ab433abd2ee8224b53ded6202cb.txt
    │           ├── meningioma_43_jpg.rf.c5fbad305e63343f4b1773e5d501adff.txt
    │           ├── meningioma_450_jpg.rf.515eabc644cd7fbbc451a9430cb0c271.txt
    │           ├── meningioma_452_jpg.rf.c18054003dbe7d88c7abd13d59e7393a.txt
    │           ├── meningioma_454_jpg.rf.9d896f5e69af0c2576a586c91b62253f.txt
    │           ├── meningioma_478_jpg.rf.842f69b991ae31cdee33ef6d08c55234.txt
    │           ├── meningioma_47_jpg.rf.0d318bff47238418444d43c002c2c905.txt
    │           ├── meningioma_496_jpg.rf.987dbb12751d93d7f893d0d37257d99c.txt
    │           ├── meningioma_499_jpg.rf.99ced13e332182c9382940d425e0e264.txt
    │           ├── meningioma_501_jpg.rf.89fc4fb7d64c4588946d015bbe4ce49c.txt
    │           ├── meningioma_503_jpg.rf.4b1927dc404e63ec1a63ae20ca889b5a.txt
    │           ├── meningioma_509_jpg.rf.8c179b9af1225f1f5ad521b9eb9ef3b3.txt
    │           ├── meningioma_516_jpg.rf.d0c5109527e9f3f694079e4e9355ac35.txt
    │           ├── meningioma_536_jpg.rf.747123a0e95af24c44d486deaf727fdc.txt
    │           ├── meningioma_549_jpg.rf.2b62f4012f34e83fdfb0ac188771e854.txt
    │           ├── meningioma_556_jpg.rf.7bccbc4cfc4d63e56ade8e3edcf93605.txt
    │           ├── meningioma_562_jpg.rf.c8a317a7417995899a34f894c9523495.txt
    │           ├── meningioma_565_jpg.rf.436f153ae0c432014185002101529a09.txt
    │           ├── meningioma_601_jpg.rf.c97e495c3e58d52ca6f9466d72f66518.txt
    │           ├── meningioma_614_jpg.rf.8b6217c6cc9fdaebaeb2771e1c71c472.txt
    │           ├── meningioma_617_jpg.rf.64d2c7b7d10a36c41ef0772efa7f062c.txt
    │           ├── meningioma_640_jpg.rf.c49efacb21a5ec7e1efec01e86923605.txt
    │           ├── meningioma_652_jpg.rf.f1a511d2df8ad06444bd0743863fb44a.txt
    │           ├── meningioma_659_jpg.rf.b5f1d60fb87bb0bddcd7488bcb7cc9bd.txt
    │           ├── meningioma_65_jpg.rf.190091d77c284bec2d2a19c2d594bbab.txt
    │           ├── meningioma_667_jpg.rf.7b3840dc68d4c7f6edb9076b107db22a.txt
    │           ├── meningioma_688_jpg.rf.e946d6c6088c00542065b3b8090ff62b.txt
    │           ├── meningioma_696_jpg.rf.e950f810702b1395b235c5cdc8f9384b.txt
    │           ├── meningioma_698_jpg.rf.fff3af97210a4990caca2d2fa374dc6a.txt
    │           ├── meningioma_708_jpg.rf.afb26cf275398fa63ffe426a34326bd2.txt
    │           ├── meningioma_712_jpg.rf.181bd43be4347f25b904af5623d21bcc.txt
    │           ├── meningioma_728_jpg.rf.50b25c5f02105fade2a41b733bd29cd2.txt
    │           ├── meningioma_735_jpg.rf.89b58d53957ab395f0a3916069c1b977.txt
    │           ├── meningioma_741_jpg.rf.7351a474636fe6a0f0b35845374a39bb.txt
    │           ├── meningioma_742_jpg.rf.593293cfccc56c9bb359c85404e113bc.txt
    │           ├── meningioma_745_jpg.rf.ff942852f698de3cedfd6569e42587a3.txt
    │           ├── meningioma_764_jpg.rf.f19fea213f59d95c947d6f8ede946a2f.txt
    │           ├── meningioma_772_jpg.rf.032d80719301deb415065e7cc2b9306e.txt
    │           ├── meningioma_789_jpg.rf.1bb9fd320e1cdf0acaf638afdc6c0955.txt
    │           ├── meningioma_79_jpg.rf.0d8ed387436af1ccbecc77149d1b098c.txt
    │           ├── meningioma_803_jpg.rf.ed6dc1faa416cc5e15683263e42dc714.txt
    │           ├── meningioma_823_jpg.rf.ba2e94412b6bd773e0aa3dd7560f2331.txt
    │           ├── meningioma_835_jpg.rf.8cdc87734332846800ce1774edfdfd5b.txt
    │           ├── meningioma_836_jpg.rf.ed2528da9d3789aa436154d02c60f7bf.txt
    │           ├── meningioma_840_jpg.rf.c0d963f29e0bee565588a01d34e0fca8.txt
    │           ├── meningioma_848_jpg.rf.eb602cae880b865a51e899ba88593e67.txt
    │           ├── meningioma_849_jpg.rf.03cac650e89f33cf6a0e1d4f57c745ce.txt
    │           ├── meningioma_860_jpg.rf.090b2389cdfec5fada884008555b1a82.txt
    │           ├── meningioma_865_jpg.rf.870d1d3447e552366f47eccf0c57a8f7.txt
    │           ├── meningioma_871_jpg.rf.3783e22c8481fa2bfd8e0ce0a8b1b24a.txt
    │           ├── meningioma_875_jpg.rf.4720e8a8acbf9c3398270b4c74aa7f0d.txt
    │           ├── meningioma_878_jpg.rf.4487f4ab0d6afba1cd025d0052a1ef67.txt
    │           ├── meningioma_884_jpg.rf.42401315a4eb27916b3b79b99e835899.txt
    │           ├── meningioma_887_jpg.rf.f4aea839efb9519489a36649a8f89b05.txt
    │           ├── meningioma_894_jpg.rf.f5e917b9f2acd2761973efb74826bcea.txt
    │           ├── meningioma_8_jpg.rf.cfef3b130d130b6cb51385d8589cc45f.txt
    │           ├── meningioma_906_jpg.rf.ba523c9e3efbfe87629e4851889cd4a8.txt
    │           ├── meningioma_908_jpg.rf.11ee7d1420dfa545edd3223c85debd32.txt
    │           ├── meningioma_911_jpg.rf.58a107efd4513c76e7e82c00e6cdf40f.txt
    │           ├── meningioma_916_jpg.rf.486f441b6c2dfcf3cb01db035ac92889.txt
    │           ├── meningioma_921_jpg.rf.8c06075a71d54ee1e4f321aef96662ab.txt
    │           ├── meningioma_92_jpg.rf.419f05266e9cec2a0cee373d87679329.txt
    │           ├── meningioma_945_jpg.rf.321c5e911f77a86199621386abaf461d.txt
    │           ├── meningioma_950_jpg.rf.76630619b029f672853d569224775be8.txt
    │           ├── meningioma_960_jpg.rf.c012468e3499961dd54bd347dca969ab.txt
    │           ├── meningioma_964_jpg.rf.dc11bf8d914d64dafa7915955dab2a0f.txt
    │           ├── meningioma_965_jpg.rf.dce38666413e6b6147fb853df960c294.txt
    │           ├── meningioma_984_jpg.rf.e983f95359483021910289163782e42a.txt
    │           ├── meningioma_991_jpg.rf.931c156f35137b76b908565975f019af.txt
    │           ├── meningioma_994_jpg.rf.9001dc4787284f16f6b1ae12e5e8ffd7.txt
    │           ├── meningioma_996_jpg.rf.60e56ad8f2e53ec66cb63d5f0bca3b50.txt
    │           ├── no_tumor_1002_jpg.rf.eb7141382aa74daadf933293752fc0e9.txt
    │           ├── no_tumor_1007_jpg.rf.353ec7f8439cfce18e6427ff90b0227d.txt
    │           ├── no_tumor_1031_jpg.rf.1813fcf1fb0b41bdb976e8d18fe1883f.txt
    │           ├── no_tumor_1065_jpg.rf.1ec3b70d96fb8b78a04d1d441e06b212.txt
    │           ├── no_tumor_1067_jpg.rf.6759483b515ce59d018e993929253c0c.txt
    │           ├── no_tumor_1081_jpg.rf.3fee39a6990fb23d44eea30539f59a86.txt
    │           ├── no_tumor_1099_jpg.rf.cd73f9b21c76fee7d8351df787a3a187.txt
    │           ├── no_tumor_1111_jpg.rf.bcdda948159a5b6c6b9692d7153b122a.txt
    │           ├── no_tumor_1130_jpg.rf.2db8f0176c51e5a76901704547e8d1c7.txt
    │           ├── no_tumor_117_jpg.rf.a77886592d46f8676d3f88224317d8bf.txt
    │           ├── no_tumor_1208_jpg.rf.98f94430f7427d8b40b37347d0a47511.txt
    │           ├── no_tumor_1213_jpg.rf.9c3c499dac7a19cc19fb3e7048a2b68d.txt
    │           ├── no_tumor_1217_jpg.rf.51b55faf0f84152be6e3badb4a17dc26.txt
    │           ├── no_tumor_1218_jpg.rf.346335503c22e7b3c561b3e653f9bf5c.txt
    │           ├── no_tumor_1223_jpg.rf.7e04a1d8bdac4b0d82a00fb230faab4e.txt
    │           ├── no_tumor_1224_jpg.rf.b32a1e11c44daa95b6dd7a247bce0915.txt
    │           ├── no_tumor_1227_jpg.rf.20a2974c7b09a97daee67d10bbef77f8.txt
    │           ├── no_tumor_1239_jpg.rf.e7f49a10e3c51a2f4716aced909a6879.txt
    │           ├── no_tumor_1263_jpg.rf.278209afdaccffa333898c2a9806d076.txt
    │           ├── no_tumor_1284_jpg.rf.65e8ae14b72a58c15f86a06096c198c9.txt
    │           ├── no_tumor_1289_jpg.rf.88f9d17ca2ac7c8cc8d7b15fc82d832a.txt
    │           ├── no_tumor_1344_jpg.rf.6997999a2a135b672fd204796d5722cb.txt
    │           ├── no_tumor_1345_jpg.rf.4f408dfee96035c8070afc955b7dfc7d.txt
    │           ├── no_tumor_1348_jpg.rf.3560eb8c558f6b4596d9556129c45b03.txt
    │           ├── no_tumor_1352_jpg.rf.03f11683c6e52a9c00f61c09aa474a26.txt
    │           ├── no_tumor_1363_jpg.rf.483ea5869df59e48dc024e3271938801.txt
    │           ├── no_tumor_136_jpg.rf.78af9cb227bc4cf835b9e27085dc118e.txt
    │           ├── no_tumor_1384_jpg.rf.2b3633eb69760763da3edb53def51fcc.txt
    │           ├── no_tumor_1386_jpg.rf.04f978ef6b137d199b27baeb69b47fd6.txt
    │           ├── no_tumor_1396_jpg.rf.164ed037bd3b4078bf95bfebfbe2f577.txt
    │           ├── no_tumor_1410_jpg.rf.4dc51a0a8672ccb40ab19e2ac0d45d24.txt
    │           ├── no_tumor_1436_jpg.rf.f9dfa062fa7fe2faad9bf6f2339bc801.txt
    │           ├── no_tumor_1446_jpg.rf.2bd824b13bed9b4682994b744ea84bd3.txt
    │           ├── no_tumor_1449_jpg.rf.23f0a67b927372d9d62fa82ae384d29d.txt
    │           ├── no_tumor_1471_jpg.rf.70f8826cb6dd5840e3f3e8cdab81914e.txt
    │           ├── no_tumor_1498_jpg.rf.0cc5de5358026d422cca78938a1b524c.txt
    │           ├── no_tumor_1516_jpg.rf.d295bf54bf757e4c11e3227514b24927.txt
    │           ├── no_tumor_1526_jpg.rf.f04364a9a0677f4c5afa9c6722332652.txt
    │           ├── no_tumor_1527_jpg.rf.48e4573bae8fb915b51adb01bbb9ee74.txt
    │           ├── no_tumor_153_jpg.rf.c20634846bf0ce46b0448de73a93b798.txt
    │           ├── no_tumor_1540_jpg.rf.f77cfa5a2c91216f53de79005f23a689.txt
    │           ├── no_tumor_1543_jpg.rf.f3ad98a16dad21108457852f884f3620.txt
    │           ├── no_tumor_1560_jpg.rf.c6bdac2fa89008814012673fe60d3508.txt
    │           ├── no_tumor_1566_jpg.rf.f57ac5bf9b8a5026a8f5ff3726afe00a.txt
    │           ├── no_tumor_1585_jpg.rf.8f7ba4467c93dde0aab4f224746fdbb5.txt
    │           ├── no_tumor_160_jpg.rf.7625f9c90ed889d0c404d10336754e20.txt
    │           ├── no_tumor_163_jpg.rf.b26bfb07fbe3a327ceee190d2a903c1b.txt
    │           ├── no_tumor_171_jpg.rf.c1116241df77b03e6d4ab77a910b4ac0.txt
    │           ├── no_tumor_183_jpg.rf.aaa5d2efc50933db68c8052329d6fd4b.txt
    │           ├── no_tumor_185_jpg.rf.e9332674f3901590135928591ee7b443.txt
    │           ├── no_tumor_194_jpg.rf.a826cac67a65f27a35b74ea688103344.txt
    │           ├── no_tumor_197_jpg.rf.d86f7d9652a73b11fa6a21d1c9cd60c5.txt
    │           ├── no_tumor_206_jpg.rf.c3e7e854c21aa7a726c844abaacfa985.txt
    │           ├── no_tumor_216_jpg.rf.81d70848b85647aaacd54e6327c689f9.txt
    │           ├── no_tumor_217_jpg.rf.8f847ed9bd96be84f764ece15d91b6e6.txt
    │           ├── no_tumor_236_jpg.rf.85c9441997ad78e26443f19c05332dd9.txt
    │           ├── no_tumor_302_jpg.rf.de592ff09d80dfcb17d95a8ede53d1e4.txt
    │           ├── no_tumor_329_jpg.rf.070fe5b5666fe2889a0516d4e93f09db.txt
    │           ├── no_tumor_32_jpg.rf.0fb9b63cf476fafec16401c5822419ea.txt
    │           ├── no_tumor_355_jpg.rf.84477c90ae6a91a6d7d46575f37742c5.txt
    │           ├── no_tumor_359_jpg.rf.572476bc6f188b37294cd551c73bff36.txt
    │           ├── no_tumor_362_jpg.rf.a97885865534be610f8dbbd58e56a432.txt
    │           ├── no_tumor_367_jpg.rf.94625e3cea0d1413b37479452bfc4db0.txt
    │           ├── no_tumor_368_jpg.rf.eb89db210e555ab0c35246d0a8c794b7.txt
    │           ├── no_tumor_370_jpg.rf.a375d8d09f8ea0274c2972816b7289d0.txt
    │           ├── no_tumor_384_jpg.rf.add95791956e5b87c5a99cbdcebc52f3.txt
    │           ├── no_tumor_388_jpg.rf.6ae31956bef019bd6b92745808f5e29c.txt
    │           ├── no_tumor_392_jpg.rf.cd8a53dff0db8fe15f7b7198b14e04bd.txt
    │           ├── no_tumor_40_jpg.rf.08020a56bb66d390b1f9061b27f9ada3.txt
    │           ├── no_tumor_41_jpg.rf.fc79c11369566f35c789381021daacea.txt
    │           ├── no_tumor_429_jpg.rf.ad8f82a83a75c8f3ea069bacb15b2667.txt
    │           ├── no_tumor_436_jpg.rf.94fb51400d8e9579600f4640fa4a1842.txt
    │           ├── no_tumor_456_jpg.rf.90e088657683fff3d66e7f5375a7599d.txt
    │           ├── no_tumor_50_jpg.rf.3be65f9c232a4652ded849dbc661cddd.txt
    │           ├── no_tumor_535_jpg.rf.a70887aedbcf0828ad42060326e941f2.txt
    │           ├── no_tumor_546_jpg.rf.cd7b37e7b4749b28f6d52c8ad6889292.txt
    │           ├── no_tumor_580_jpg.rf.a6973334fe566d120d5e89e43012ee12.txt
    │           ├── no_tumor_613_jpg.rf.03e2f7f23767650797806734a2466921.txt
    │           ├── no_tumor_615_jpg.rf.69a773dbaf49fcd6b71f890fdb8fc5e7.txt
    │           ├── no_tumor_622_jpg.rf.4ec6d2dee2d9178e74a0031e87d9d48b.txt
    │           ├── no_tumor_62_jpg.rf.ef72cd835b012e3ecc8829d4a0fae30d.txt
    │           ├── no_tumor_639_jpg.rf.9a390098e900f70cc25290f5c2b2acfc.txt
    │           ├── no_tumor_642_jpg.rf.512324112aae0a5ec37a1d955b6ec2d9.txt
    │           ├── no_tumor_643_jpg.rf.73be8e13c31c16c3f3de6d73d81f5a41.txt
    │           ├── no_tumor_64_jpg.rf.c37f888a9cd7193a8282b28ad1e821b1.txt
    │           ├── no_tumor_65_jpg.rf.d0c617db44ca384b082d71b7bf4407c0.txt
    │           ├── no_tumor_665_jpg.rf.be649898c2cdde1292dc507e28da0927.txt
    │           ├── no_tumor_675_jpg.rf.1ad1f8846e1d8537cb1f0d83afce820d.txt
    │           ├── no_tumor_677_jpg.rf.c3e61ad4137f047ad843e240c03858c8.txt
    │           ├── no_tumor_683_jpg.rf.c44df82d787ccf30e1d50529d8beb588.txt
    │           ├── no_tumor_741_jpg.rf.8295ca43d5df8144af439d0869af0881.txt
    │           ├── no_tumor_771_jpg.rf.e40daaf725f1f6817dfeeeff5c66a145.txt
    │           ├── no_tumor_777_jpg.rf.6d02a1ac2cd70272de1cc347ff43fa0e.txt
    │           ├── no_tumor_778_jpg.rf.85b0a33e447c7fb1160ae25e6f255cbe.txt
    │           ├── no_tumor_782_jpg.rf.82c6b020ac0f81fe14e28a880febfec3.txt
    │           ├── no_tumor_789_jpg.rf.7bbf779a54237b93b2a590d5de20ee59.txt
    │           ├── no_tumor_798_jpg.rf.d1d5b7d9bd8c9372eae7268d7d74f60a.txt
    │           ├── no_tumor_803_jpg.rf.7e7b6949454153f02fece110715ba2bf.txt
    │           ├── no_tumor_820_jpg.rf.09b1e08174791ce390a1d64357d82f1d.txt
    │           ├── no_tumor_844_jpg.rf.5c4388e5461f6ba4346218e13e6ed9a8.txt
    │           ├── no_tumor_84_jpg.rf.e0c80858940cf7d4e12b89756f9f71b3.txt
    │           ├── no_tumor_854_jpg.rf.3753a0769e4956bf721130cd6f9caa83.txt
    │           ├── no_tumor_855_jpg.rf.a52d1a3997edfe6d0d665a98b8a88694.txt
    │           ├── no_tumor_890_jpg.rf.eb4da32fb507e2296f266fc017910798.txt
    │           ├── no_tumor_892_jpg.rf.895906826facdd5d906e9f997092d56d.txt
    │           ├── no_tumor_905_jpg.rf.1c4afb0f72368adba5f7fdb315c90e62.txt
    │           ├── no_tumor_907_jpg.rf.74a36007d3ef39b6458b7ddee7582e1b.txt
    │           ├── no_tumor_951_jpg.rf.1176685798923047beda3f04c8486b0c.txt
    │           ├── no_tumor_956_jpg.rf.320de10ee7815a41f6c73642dc7bc722.txt
    │           ├── no_tumor_957_jpg.rf.449e467036fb02ccb06154af2e28e7f7.txt
    │           ├── no_tumor_966_jpg.rf.46c7980a932d6269c724c002a73fe69a.txt
    │           ├── no_tumor_969_jpg.rf.c177054ee4015cf9d5171c5618e3ef6d.txt
    │           ├── no_tumor_995_jpg.rf.b3de5dbbc16cdd0204f064fe483c1c37.txt
    │           ├── no_tumor_996_jpg.rf.408890d2c8475c8c401a82afec20f035.txt
    │           ├── no_tumor_999_jpg.rf.d1687a97dc2837381fbe3a7371f3c99c.txt
    │           ├── pituitary_1009_jpg.rf.e179a6d77c8a0ce652c26b7aee667ceb.txt
    │           ├── pituitary_1026_jpg.rf.fa615e9c25e6aa3a263f679407b4a1a5.txt
    │           ├── pituitary_1044_jpg.rf.19d7eda33009adc9b1db6bd9083d8b98.txt
    │           ├── pituitary_1067_jpg.rf.0c9002bc461810e986834b904ca24f6e.txt
    │           ├── pituitary_1076_jpg.rf.6a8c3f73b891ffe6f81cf8e03582dafc.txt
    │           ├── pituitary_1092_jpg.rf.1d6a0ba11712209997009f547c16d238.txt
    │           ├── pituitary_1098_jpg.rf.d91b94211cbec86e864f47fd77a81eb3.txt
    │           ├── pituitary_1104_jpg.rf.77a4b9e4245dc05bc100a23e1cd0acb9.txt
    │           ├── pituitary_1107_jpg.rf.f59ef80dcd2a4136de36124616bc435c.txt
    │           ├── pituitary_1108_jpg.rf.a7bf774d2fb15ba62f47d79f33b74a24.txt
    │           ├── pituitary_1121_jpg.rf.b4704ff58bc42db83bf464df9c19e75b.txt
    │           ├── pituitary_1127_jpg.rf.aeffce333eb2c5de56ffdb1d587c0369.txt
    │           ├── pituitary_1152_jpg.rf.e47bd8fa3a9c1e4f829b69cb4375d7d8.txt
    │           ├── pituitary_1165_jpg.rf.94e81c2cb82675782dc4e5921be405ab.txt
    │           ├── pituitary_1171_jpg.rf.c6eef3d6c3ec7831fbcf0fc0aca4fb3b.txt
    │           ├── pituitary_1180_jpg.rf.ae6aa98e15f082e177ac2090c74ef02b.txt
    │           ├── pituitary_1197_jpg.rf.07965526c920bba41c1d5b7dd0089f1f.txt
    │           ├── pituitary_1198_jpg.rf.11d1c54285218a5b1dde50d508cfc794.txt
    │           ├── pituitary_1208_jpg.rf.f6af8616dfe73e7f284d9db3de136782.txt
    │           ├── pituitary_1222_jpg.rf.9560cfc6b55c48f9aa1692716de4e245.txt
    │           ├── pituitary_1247_jpg.rf.0f885c770dfeadef6da09d14a0aec1b9.txt
    │           ├── pituitary_1260_jpg.rf.962d995dae3c5e32c701d1b778f74d41.txt
    │           ├── pituitary_1262_jpg.rf.467185be0f99bf9b3d421767f440bf2b.txt
    │           ├── pituitary_1270_jpg.rf.625c288b1b6b7e19298316b10041e618.txt
    │           ├── pituitary_1293_jpg.rf.227ec5924fa283fc25b48a0b5ddcd1f9.txt
    │           ├── pituitary_1295_jpg.rf.a63887af4973fc134e57b1af4e60c675.txt
    │           ├── pituitary_1315_jpg.rf.0aebcf4347718666c98f20beea90aa41.txt
    │           ├── pituitary_1330_jpg.rf.7d13fca1123687a5db40fceeb350552a.txt
    │           ├── pituitary_1345_jpg.rf.537cc53b8da80358a6a661da0d61d3d5.txt
    │           ├── pituitary_1350_jpg.rf.98e44dd4b49fc352d659f065614ad9a0.txt
    │           ├── pituitary_1351_jpg.rf.ee8fe8c699b2b99e01c7f41d69410e98.txt
    │           ├── pituitary_1376_jpg.rf.36043ee2f3fc026f1ae2062299c0b505.txt
    │           ├── pituitary_1380_jpg.rf.d2cddf63eb15ae086dba6d9296236f71.txt
    │           ├── pituitary_1386_jpg.rf.b4cdf1145d81200512ff6d610473eceb.txt
    │           ├── pituitary_139_jpg.rf.6b793346b7668ed1f42c419b54999916.txt
    │           ├── pituitary_1437_jpg.rf.f58b5bec741ad04c78c44c2c5e2c430a.txt
    │           ├── pituitary_1440_jpg.rf.914705ce7e047c1b93557f9724dd6a4e.txt
    │           ├── pituitary_1448_jpg.rf.11d6d3135dfcef91d320133d2b2fd2e3.txt
    │           ├── pituitary_1449_jpg.rf.a391cc17ec28a458bf5d8e86a0ea1b8f.txt
    │           ├── pituitary_1456_jpg.rf.65fea9431b09018b4afdc9e26a3102ec.txt
    │           ├── pituitary_156_jpg.rf.169d34a242c1aab392535e0612a1fbd5.txt
    │           ├── pituitary_171_jpg.rf.6290ddd1131aafcb4bce1167db5f19ea.txt
    │           ├── pituitary_189_jpg.rf.649a81cb433ea19e0f07aa8b00f2554d.txt
    │           ├── pituitary_197_jpg.rf.39d697f7e73591c428e6546b826ac68f.txt
    │           ├── pituitary_207_jpg.rf.e268e6d86155a9c9017e1b816eea0eef.txt
    │           ├── pituitary_214_jpg.rf.b4ada692026dbea30c65e447f3430d69.txt
    │           ├── pituitary_259_jpg.rf.e7a8af82dc71e9fcc2edb0a15ec476c4.txt
    │           ├── pituitary_263_jpg.rf.915b29f4bfea50f135f1647859387a5f.txt
    │           ├── pituitary_288_jpg.rf.e802e126c7930bed0d6d3d7a0b83e900.txt
    │           ├── pituitary_311_jpg.rf.1204ad349fb21342fb37d4aaa6130976.txt
    │           ├── pituitary_338_jpg.rf.eeeb860df08888c687e8ffead7165ca1.txt
    │           ├── pituitary_34_jpg.rf.dd47d9c32ae228342cda345fd308c1ca.txt
    │           ├── pituitary_354_jpg.rf.9d4d3a32f68c95b22b9e370a0b4efa8d.txt
    │           ├── pituitary_36_jpg.rf.fd6705c944dab1fb3962524d35ddcf2f.txt
    │           ├── pituitary_403_jpg.rf.d573650d6ff7a0154fc704d7607ef9a5.txt
    │           ├── pituitary_404_jpg.rf.677d57ac4ee3c3fb96ce4efe22e9063f.txt
    │           ├── pituitary_411_jpg.rf.748049333a6fce2222cf7511835be3fb.txt
    │           ├── pituitary_470_jpg.rf.f1f9b1c6cfca246fb2d1a8cc708a417d.txt
    │           ├── pituitary_472_jpg.rf.d78e5d88f668fdb2d584e8cbc431c106.txt
    │           ├── pituitary_475_jpg.rf.81c6fdd17bdff50b43b76b75a97e3543.txt
    │           ├── pituitary_493_jpg.rf.37401d79404f9906aaa99e85f7c88fe2.txt
    │           ├── pituitary_497_jpg.rf.aaef3a2853a5dd9e8309eca35811b527.txt
    │           ├── pituitary_533_jpg.rf.92c341f51cb080a7059811307a860796.txt
    │           ├── pituitary_542_jpg.rf.c739efdaa53527636057bc1fcb68526c.txt
    │           ├── pituitary_546_jpg.rf.72f5354a69759a3b37185b519178955d.txt
    │           ├── pituitary_565_jpg.rf.d6cf0d552749c62f69b1cf1d3bc4d12c.txt
    │           ├── pituitary_567_jpg.rf.bb5b6ca6391cac363f1a1a69d33c07b5.txt
    │           ├── pituitary_589_jpg.rf.ef5f8bc3b105320502868193ab191c87.txt
    │           ├── pituitary_60_jpg.rf.1a4b4517ce496f69b86facecb4296981.txt
    │           ├── pituitary_626_jpg.rf.a1620fbfeae1bf27ab8d1e35ca5f321f.txt
    │           ├── pituitary_639_jpg.rf.acc1152352e1d325b776d5728c807413.txt
    │           ├── pituitary_640_jpg.rf.e4c12aca7b7c313222602aebf557be0d.txt
    │           ├── pituitary_649_jpg.rf.7e31ac2655ed36becd74b70a2e8cc828.txt
    │           ├── pituitary_657_jpg.rf.9a5199697d475febd4cfe5f0056a66dc.txt
    │           ├── pituitary_692_jpg.rf.9029e220dbb8f8bb70ce68e8a83c806f.txt
    │           ├── pituitary_698_jpg.rf.d435e3f15b867ebf52a22a63aaea5a5d.txt
    │           ├── pituitary_700_jpg.rf.a321c78a245b0d7fa88bd88befa55ccb.txt
    │           ├── pituitary_705_jpg.rf.5937ed706720a93c4b3420ed66df3e54.txt
    │           ├── pituitary_710_jpg.rf.289de300deb1a862a2afb4354da1f7fc.txt
    │           ├── pituitary_721_jpg.rf.9fd4133fe7ecaa1e04eafd2005a1acf2.txt
    │           ├── pituitary_733_jpg.rf.31501bcc82220ca2f87df8d1ccc85da2.txt
    │           ├── pituitary_735_jpg.rf.8a8dd2f2ac0c494dc0aae8398042c87f.txt
    │           ├── pituitary_742_jpg.rf.d607bca9083caa3adad6e52ccd02d54c.txt
    │           ├── pituitary_746_jpg.rf.cdd8dbf8f485107bd5c72e7ac134ce13.txt
    │           ├── pituitary_748_jpg.rf.747584aa30b5d6e4463549cf7e44fe74.txt
    │           ├── pituitary_752_jpg.rf.68e1176f678b63c866968b1a53dd0a16.txt
    │           ├── pituitary_774_jpg.rf.edd6e4ca6dcef76261f4ed6f57264e2f.txt
    │           ├── pituitary_775_jpg.rf.136418ce384ae4b66c20095012be9481.txt
    │           ├── pituitary_777_jpg.rf.e6602fb2d738e4c622b2a7fa4b569073.txt
    │           ├── pituitary_781_jpg.rf.5343e96691d37277402dafb1a48ead92.txt
    │           ├── pituitary_782_jpg.rf.7827aa6732810e05fe211ce5e688cb01.txt
    │           ├── pituitary_785_jpg.rf.f221f95749e78883efbf98b858af826b.txt
    │           ├── pituitary_7_jpg.rf.b42b94a81cba4fc6ff7c6de77efcfe2a.txt
    │           ├── pituitary_805_jpg.rf.695fc45c8441c4e2f33286a3c0954637.txt
    │           ├── pituitary_818_jpg.rf.aef4102ca82d145dcf9797b039db2c82.txt
    │           ├── pituitary_828_jpg.rf.348b0e189d063776ac400fd9ea5fde9a.txt
    │           ├── pituitary_878_jpg.rf.208922f4eef293b44c8ab463033157a6.txt
    │           ├── pituitary_885_jpg.rf.f255cf825b37083846017c6d4d24b9bd.txt
    │           ├── pituitary_889_jpg.rf.2c119b815694de488c965a9cdfa23261.txt
    │           ├── pituitary_910_jpg.rf.f12685eb37f15d08ce604a496f526b3d.txt
    │           ├── pituitary_911_jpg.rf.4f4db29616e6d538754dc7b63191190e.txt
    │           ├── pituitary_912_jpg.rf.8ee8f5860ad33785a7c479732467305a.txt
    │           ├── pituitary_917_jpg.rf.592ab4188f0cec2e64fc2e75213cef11.txt
    │           ├── pituitary_933_jpg.rf.365b22a61ed488ab4dc23230d772900e.txt
    │           ├── pituitary_937_jpg.rf.e4d8a601c65d84971df3794dd86bba7e.txt
    │           └── pituitary_990_jpg.rf.33995d119eb8785c89eaef7e652c2ca2.txt
    └── .ipynb_checkpoints/


================================================
FILE: READEME.md.txt
================================================
Brain Tumor Segmentation with YOLOv11 and SAM2

This project uses the YOLOv11 object detection model combined with SAM2 (Segment Anything Model) to detect and segment brain tumors from MRI images. The goal is to provide both classification (tumor type) and pixel-wise segmentation masks for visual interpretation.

📁 Dataset

Source: Roboflow Tumor Detection Dataset (v8)

Contains four tumor classes:

Glioma

Meningioma

Pituitary

No Tumor

⚙️ Environment Setup

Note: This project was run on CPU, not CUDA/GPU.

1. Install PyTorch (CPU)

Visit: https://pytorch.org/get-started/locally

Or install directly:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

2. Install Ultralytics

pip install ultralytics

🚀 Running the Code

Load YOLO model (already trained on tumor dataset):

from ultralytics import YOLO
model = YOLO("path/to/best.pt")

Run predictions:

results = model("path/to/images")

Load SAM2 and apply segmentation masks:

from ultralytics import SAM
sam_model = SAM("sam2_b.pt")

for i, result in enumerate(results):
    boxes = result.boxes.xyxy
    sam_results_list = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device="cpu")
    for j, sam_result in enumerate(sam_results_list):
        mask = sam_result.masks.data[0].cpu().numpy().astype('uint8') * 255
        # Save or process mask

🧠 Output

Classification and bounding boxes from YOLOv11

Pixel-level masks from SAM2

Segmented images are saved to the working directory

📝 Author

Syed Daniyal Haider NaqviComputer Science Student — COMSATS University Islamabad

🔗 License

This project is for academic and research purposes.




================================================
FILE: yolo11n.pt
================================================
[Non-text file]





================================================
FILE: TumorDetection/data.yaml
================================================
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['NO_tumor', 'glioma', 'meningioma', 'pituitary', 'space-occupying lesion-']

roboflow:
  workspace: brain-tumor-detection-wsera
  project: tumor-detection-ko5jp
  version: 8
  license: CC BY 4.0
  url: https://universe.roboflow.com/brain-tumor-detection-wsera/tumor-detection-ko5jp/dataset/8


================================================
FILE: TumorDetection/README.roboflow.txt
================================================

Tumor Detection - v8 2024-07-31 2:19pm
==============================

This dataset was exported via roboflow.com on October 11, 2024 at 3:42 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 1956 images.
Glioma-meningioma-pituitary-No are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

No image augmentation techniques were applied.







================================================
FILE: TumorDetection/train/labels/glioma_102_jpg.rf.e184590079f7726ff64daccd83d8ff99.txt
================================================
1 0.4150390625 0.47265625 0.41015625 0.4755859375 0.41796875 0.4677734375 0.41015625 0.4560546875 0.416015625 0.4521484375 0.40625 0.4482421875 0.408203125 0.4384765625 0.400390625 0.4326171875 0.40625 0.4208984375 0.3837890625 0.41796875 0.376953125 0.4091796875 0.384765625 0.3974609375 0.3798828125 0.375 0.3291015625 0.3671875 0.30859375 0.3857421875 0.3125 0.4169921875 0.3193359375 0.419921875 0.3232421875 0.412109375 0.3310546875 0.421875 0.3388671875 0.416015625 0.35546875 0.4228515625 0.3505859375 0.431640625 0.3408203125 0.423828125 0.3271484375 0.43359375 0.30859375 0.4248046875 0.3125 0.4482421875 0.302734375 0.5048828125 0.3125 0.5126953125 0.3154296875 0.47265625 0.32421875 0.4970703125 0.3203125 0.5087890625 0.326171875 0.5126953125 0.3291015625 0.4921875 0.333984375 0.5244140625 0.3486328125 0.525390625 0.35546875 0.5458984375 0.37109375 0.5478515625 0.40234375 0.5341796875 0.404296875 0.5009765625 0.41796875 0.4814453125 0.4150390625 0.47265625


================================================
FILE: TumorDetection/train/labels/glioma_1086_jpg.rf.a2785a388f7efbd5f665601ec9147d36.txt
================================================
1 0.62109375 0.4794921875 0.6044921875 0.458984375 0.5439453125 0.4453125 0.55078125 0.4873046875 0.54296875 0.5029296875 0.5478515625 0.498046875 0.5634765625 0.521484375 0.58203125 0.5224609375 0.5849609375 0.509765625 0.6044921875 0.509765625 0.6171875 0.4990234375 0.62109375 0.4794921875


================================================
FILE: TumorDetection/train/labels/glioma_1112_jpg.rf.8f37ed9f604563cb38cb6f32ce3acf16.txt
================================================
1 0.765625 0.3232421875 0.759765625 0.3056640625 0.7275390625 0.275390625 0.716796875 0.2744140625 0.7177734375 0.265625 0.724609375 0.2724609375 0.7216796875 0.263671875 0.6923828125 0.2421875 0.6689453125 0.2421875 0.6318359375 0.2578125 0.603515625 0.3056640625 0.6220703125 0.328125 0.6455078125 0.32421875 0.7158203125 0.359375 0.7412109375 0.328125 0.75 0.3291015625 0.75 0.3388671875 0.736328125 0.3466796875 0.75 0.3662109375 0.7353515625 0.359375 0.734375 0.3701171875 0.7177734375 0.37109375 0.7265625 0.3779296875 0.7099609375 0.38671875 0.6650390625 0.38671875 0.681640625 0.3916015625 0.6640625 0.3974609375 0.6787109375 0.40234375 0.654296875 0.3994140625 0.658203125 0.3583984375 0.6533203125 0.3515625 0.63671875 0.3564453125 0.650390625 0.3623046875 0.640625 0.3740234375 0.65234375 0.3779296875 0.6474609375 0.38671875 0.6328125 0.3896484375 0.65234375 0.3935546875 0.630859375 0.3994140625 0.6689453125 0.41796875 0.69140625 0.4189453125 0.7119140625 0.416015625 0.73828125 0.3955078125 0.763671875 0.3486328125 0.765625 0.3232421875


================================================
FILE: TumorDetection/train/labels/glioma_1133_jpg.rf.8f564b9043b1ec8362fc38df344bbe1e.txt
================================================
1 0.5712890625 0.42578125 0.5390625 0.4462890625 0.5234375 0.4912109375 0.5576171875 0.53125 0.60546875 0.5341796875 0.634765625 0.4775390625 0.62890625 0.4482421875 0.5986328125 0.42578125 0.5712890625 0.42578125


================================================
FILE: TumorDetection/train/labels/glioma_1138_jpg.rf.c6a259e4343f9bfb52cfa69eb60f46c7.txt
================================================
1 0.7353515625 0.572265625 0.71484375 0.5654296875 0.724609375 0.5458984375 0.7177734375 0.5390625 0.6875 0.5478515625 0.69140625 0.5751953125 0.6845703125 0.587890625 0.681640625 0.5693359375 0.65625 0.5498046875 0.671875 0.4931640625 0.6552734375 0.46875 0.64453125 0.4716796875 0.6298828125 0.50390625 0.5849609375 0.521484375 0.5498046875 0.515625 0.51953125 0.5439453125 0.521484375 0.5615234375 0.5458984375 0.572265625 0.5595703125 0.591796875 0.5927734375 0.599609375 0.5859375 0.5966796875 0.5947265625 0.587890625 0.595703125 0.6005859375 0.6259765625 0.62109375 0.63671875 0.6474609375 0.646484375 0.6494140625 0.6728515625 0.646484375 0.7216796875 0.615234375 0.740234375 0.5869140625 0.7353515625 0.572265625


================================================
FILE: TumorDetection/train/labels/glioma_1147_jpg.rf.e1dfee64ef3faccf1c0ace058d230551.txt
================================================
1 0.3662109375 0.56640625 0.359375 0.5693359375 0.36328125 0.5478515625 0.3564453125 0.53515625 0.357421875 0.5654296875 0.33203125 0.6005859375 0.33984375 0.6708984375 0.3544921875 0.66796875 0.3759765625 0.6796875 0.4267578125 0.673828125 0.439453125 0.6787109375 0.416015625 0.6025390625 0.3759765625 0.564453125 0.3681640625 0.572265625 0.3662109375 0.56640625


================================================
FILE: TumorDetection/train/labels/glioma_1202_jpg.rf.24044ec3eab86394cd061a4a8735a0d9.txt
================================================
1 0.5595703125 0.43359375 0.556640625 0.4443359375 0.568359375 0.4638671875 0.576171875 0.5029296875 0.6083984375 0.53515625 0.626953125 0.5361328125 0.677734375 0.5087890625 0.666015625 0.4541015625 0.6025390625 0.421875 0.5595703125 0.43359375


================================================
FILE: TumorDetection/train/labels/glioma_120_jpg.rf.7103603cabccecd5d6c0a7d875b2b5d7.txt
================================================
1 0.4794921875 0.3828125 0.4697265625 0.390625 0.466796875 0.4150390625 0.474609375 0.4326171875 0.498046875 0.4404296875 0.5458984375 0.421875 0.544921875 0.4013671875 0.5107421875 0.3828125 0.4794921875 0.3828125
1 0.7177734375 0.58984375 0.75 0.6650390625 0.76171875 0.6455078125 0.759765625 0.6181640625 0.7431640625 0.595703125 0.7177734375 0.58984375


================================================
FILE: TumorDetection/train/labels/glioma_1228_jpg.rf.068c264838666499bca0d7060de33af9.txt
================================================
1 0.4140625 0.5517578125 0.423828125 0.5517578125 0.443359375 0.5361328125 0.4384765625 0.5234375 0.408203125 0.5263671875 0.4140625 0.5517578125


================================================
FILE: TumorDetection/train/labels/glioma_1235_jpg.rf.2a8808fa4ef7eeca9ee3b039dd233373.txt
================================================
1 0.4755859375 0.255859375 0.4541015625 0.2578125 0.4453125 0.2822265625 0.447265625 0.3173828125 0.49609375 0.3408203125 0.505859375 0.3271484375 0.486328125 0.2958984375 0.490234375 0.2646484375 0.4755859375 0.255859375


================================================
FILE: TumorDetection/train/labels/glioma_1240_jpg.rf.f9ce263457e8d2b30da8d2876f8ac132.txt
================================================
1 0.4755859375 0.259765625 0.4462890625 0.26171875 0.4296875 0.2939453125 0.431640625 0.3154296875 0.4482421875 0.33984375 0.490234375 0.3505859375 0.515625 0.3330078125 0.490234375 0.3017578125 0.48828125 0.2705078125 0.4755859375 0.259765625


================================================
FILE: TumorDetection/train/labels/glioma_1266_jpg.rf.8d9c69dba00e64fcf859ffe1b8f34ed7.txt
================================================
1 0.5185546875 0.41796875 0.4814453125 0.404296875 0.3974609375 0.396484375 0.3564453125 0.375 0.2822265625 0.39453125 0.24609375 0.4208984375 0.240234375 0.4482421875 0.2509765625 0.462890625 0.3115234375 0.474609375 0.3798828125 0.4453125 0.373046875 0.4814453125 0.3994140625 0.51171875 0.4873046875 0.541015625 0.509765625 0.5419921875 0.5302734375 0.537109375 0.552734375 0.4990234375 0.546875 0.4462890625 0.5185546875 0.41796875


================================================
FILE: TumorDetection/train/labels/glioma_1272_jpg.rf.634083758e76317aecd2f06cf736ebb4.txt
================================================
1 0.6689453125 0.4375 0.662109375 0.4443359375 0.666015625 0.4736328125 0.6806640625 0.48828125 0.6953125 0.4892578125 0.71484375 0.4697265625 0.7109375 0.4482421875 0.6962890625 0.435546875 0.6689453125 0.4375


================================================
FILE: TumorDetection/train/labels/glioma_128_jpg.rf.b238fdefc66ce62f90c11aa4ff7cd10f.txt
================================================
1 0.22265625 0.4912109375 0.2236328125 0.51171875 0.2451171875 0.52734375 0.265625 0.5263671875 0.28125 0.5146484375 0.2666015625 0.494140625 0.2412109375 0.482421875 0.22265625 0.4912109375


================================================
FILE: TumorDetection/train/labels/glioma_1296_jpg.rf.c0e3188a4d00bfba320b904486a84794.txt
================================================
1 0.5439453125 0.67578125 0.521484375 0.6435546875 0.5234375 0.5966796875 0.5068359375 0.55859375 0.494140625 0.6748046875 0.4990234375 0.6875 0.5107421875 0.6796875 0.515625 0.6962890625 0.521484375 0.6962890625 0.5283203125 0.68359375 0.5478515625 0.693359375 0.572265625 0.6884765625 0.5439453125 0.67578125


================================================
FILE: TumorDetection/train/labels/glioma_1301_jpg.rf.d70f5b0990be3d5ca5d2359623d84a90.txt
================================================
1 0.486328125 0.4033203125 0.50390625 0.4091796875 0.513671875 0.3916015625 0.4990234375 0.380859375 0.486328125 0.3896484375 0.486328125 0.4033203125
1 0.48828125 0.6142578125 0.486328125 0.6806640625 0.498046875 0.6845703125 0.515625 0.6103515625 0.5068359375 0.59375 0.482421875 0.6044921875 0.48828125 0.6142578125


================================================
FILE: TumorDetection/train/labels/glioma_145_jpg.rf.9125a688625012628983f59e080c1986.txt
================================================
1 0.48046875 0.4013671875 0.466796875 0.3935546875 0.47265625 0.3837890625 0.46484375 0.3623046875 0.4462890625 0.345703125 0.3427734375 0.310546875 0.2978515625 0.318359375 0.2939453125 0.306640625 0.28515625 0.3232421875 0.29296875 0.3271484375 0.2734375 0.3447265625 0.267578125 0.4013671875 0.2939453125 0.390625 0.298828125 0.4052734375 0.287109375 0.4248046875 0.2890625 0.4404296875 0.2705078125 0.44140625 0.265625 0.4521484375 0.2705078125 0.4609375 0.2978515625 0.46875 0.3330078125 0.5078125 0.375 0.5146484375 0.4384765625 0.505859375 0.4609375 0.4912109375 0.482421875 0.4365234375 0.48046875 0.4013671875


================================================
FILE: TumorDetection/train/labels/glioma_171_jpg.rf.ca11a72a9de2a480ff956b4306c2ea45.txt
================================================
1 0.6416015625 0.40234375 0.5947265625 0.392578125 0.5361328125 0.41015625 0.513671875 0.4541015625 0.521484375 0.4892578125 0.533203125 0.5068359375 0.53125 0.5244140625 0.5576171875 0.53515625 0.6142578125 0.525390625 0.609375 0.5556640625 0.6171875 0.5595703125 0.66015625 0.4990234375 0.662109375 0.4560546875 0.65234375 0.4150390625 0.6416015625 0.40234375


================================================
FILE: TumorDetection/train/labels/glioma_172_jpg.rf.2456b08df0d238e939af32d2a0f31ae8.txt
================================================
1 0.5927734375 0.416015625 0.5712890625 0.412109375 0.529296875 0.4462890625 0.53515625 0.4716796875 0.548828125 0.4833984375 0.5390625 0.4970703125 0.5517578125 0.505859375 0.6015625 0.5146484375 0.6181640625 0.5078125 0.63671875 0.4814453125 0.630859375 0.4501953125 0.5927734375 0.416015625


================================================
FILE: TumorDetection/train/labels/glioma_191_jpg.rf.77be681637a02f58673ed32218116004.txt
================================================
1 0.5771484375 0.5 0.5595703125 0.494140625 0.552734375 0.5126953125 0.5546875 0.5439453125 0.568359375 0.5458984375 0.591796875 0.5166015625 0.578125 0.5126953125 0.5810546875 0.494140625 0.5771484375 0.5
1 0.470703125 0.4931640625 0.466796875 0.4677734375 0.4482421875 0.462890625 0.443359375 0.4677734375 0.453125 0.4716796875 0.4501953125 0.484375 0.4609375 0.4716796875 0.453125 0.4853515625 0.458984375 0.4873046875 0.453125 0.5029296875 0.458984375 0.5029296875 0.455078125 0.5341796875 0.4609375 0.5458984375 0.484375 0.5283203125 0.484375 0.5009765625 0.4775390625 0.48828125 0.470703125 0.4931640625
1 0.4326171875 0.552734375 0.4375 0.5478515625 0.412109375 0.5029296875 0.4150390625 0.482421875 0.3662109375 0.501953125 0.353515625 0.5146484375 0.345703125 0.5537109375 0.35546875 0.5654296875 0.359375 0.5302734375 0.3955078125 0.51171875 0.4248046875 0.546875 0.4326171875 0.552734375


================================================
FILE: TumorDetection/train/labels/glioma_22_jpg.rf.991216b0b4606e74817673a45b166a34.txt
================================================
1 0.5654296875 0.271484375 0.541015625 0.3056640625 0.54296875 0.3408203125 0.5615234375 0.373046875 0.60546875 0.3857421875 0.630859375 0.3662109375 0.626953125 0.3076171875 0.6005859375 0.27734375 0.5654296875 0.271484375


================================================
FILE: TumorDetection/train/labels/glioma_238_jpg.rf.51666b7b153f1e603cf1558cf99f7e7d.txt
================================================
1 0.486328125 0.3232421875 0.4736328125 0.306640625 0.4150390625 0.31640625 0.3662109375 0.34765625 0.33203125 0.3876953125 0.322265625 0.4287109375 0.30078125 0.4443359375 0.3076171875 0.453125 0.3486328125 0.45703125 0.353515625 0.4697265625 0.3779296875 0.486328125 0.400390625 0.4873046875 0.4619140625 0.4765625 0.48046875 0.4580078125 0.490234375 0.3759765625 0.486328125 0.3232421875


================================================
FILE: TumorDetection/train/labels/glioma_259_jpg.rf.60db168e3d887d8db6795dd989766381.txt
================================================
1 0.5361328125 0.462890625 0.5185546875 0.4765625 0.4990234375 0.478515625 0.4716796875 0.443359375 0.4609375 0.4736328125 0.4375 0.5009765625 0.4697265625 0.50390625 0.53515625 0.5263671875 0.548828125 0.5146484375 0.54296875 0.4951171875 0.546875 0.4677734375 0.5361328125 0.462890625


================================================
FILE: TumorDetection/train/labels/glioma_283_jpg.rf.d2e24030d360608bbb0038c7073d7faf.txt
================================================
1 0.6015625 0.6201171875 0.615234375 0.6123046875 0.6171875 0.5908203125 0.6123046875 0.580078125 0.6015625 0.5830078125 0.5947265625 0.5625 0.583984375 0.5595703125 0.578125 0.5029296875 0.5517578125 0.4921875 0.5361328125 0.498046875 0.5302734375 0.484375 0.5146484375 0.484375 0.5126953125 0.498046875 0.4853515625 0.494140625 0.4921875 0.4970703125 0.4912109375 0.5078125 0.4833984375 0.5078125 0.4833984375 0.5 0.4716796875 0.509765625 0.4638671875 0.501953125 0.4580078125 0.515625 0.4453125 0.5166015625 0.447265625 0.5263671875 0.4296875 0.5341796875 0.435546875 0.5517578125 0.40625 0.5908203125 0.404296875 0.6083984375 0.4140625 0.6181640625 0.41015625 0.6416015625 0.4345703125 0.67578125 0.4404296875 0.6640625 0.4521484375 0.671875 0.4599609375 0.6640625 0.4775390625 0.70703125 0.4912109375 0.701171875 0.55078125 0.7158203125 0.58203125 0.6767578125 0.58203125 0.6630859375 0.607421875 0.6220703125 0.6015625 0.6201171875


================================================
FILE: TumorDetection/train/labels/glioma_31_jpg.rf.f88f3b4e4a83554f8bde206b1b9bce4f.txt
================================================
1 0.4658203125 0.216796875 0.44921875 0.2451171875 0.484375 0.2568359375 0.501953125 0.2470703125 0.51171875 0.2255859375 0.4892578125 0.21484375 0.4658203125 0.216796875
1 0.5947265625 0.287109375 0.5419921875 0.271484375 0.5185546875 0.279296875 0.490234375 0.3037109375 0.474609375 0.3349609375 0.470703125 0.3642578125 0.4853515625 0.390625 0.50390625 0.3935546875 0.5263671875 0.384765625 0.572265625 0.3427734375 0.595703125 0.3037109375 0.5947265625 0.287109375
1 0.6669921875 0.453125 0.6923828125 0.451171875 0.705078125 0.4345703125 0.7109375 0.3955078125 0.7021484375 0.38671875 0.6474609375 0.388671875 0.642578125 0.3681640625 0.6259765625 0.361328125 0.59765625 0.3916015625 0.5869140625 0.435546875 0.5693359375 0.44140625 0.5537109375 0.4609375 0.5009765625 0.49609375 0.4697265625 0.49609375 0.439453125 0.5087890625 0.46484375 0.5107421875 0.5087890625 0.5078125 0.5673828125 0.470703125 0.6396484375 0.466796875 0.6669921875 0.453125


================================================
FILE: TumorDetection/train/labels/glioma_322_jpg.rf.0fa79c110072428a02bb441c57901d3f.txt
================================================
1 0.529296875 0.4521484375 0.5166015625 0.4453125 0.5068359375 0.44921875 0.505859375 0.4775390625 0.53125 0.5302734375 0.619140625 0.6240234375 0.623046875 0.6123046875 0.615234375 0.5908203125 0.57421875 0.5400390625 0.529296875 0.4521484375
1 0.4697265625 0.462890625 0.451171875 0.4677734375 0.44921875 0.4951171875 0.423828125 0.5244140625 0.42578125 0.5498046875 0.404296875 0.6083984375 0.40625 0.6396484375 0.4296875 0.6201171875 0.44921875 0.5615234375 0.46484375 0.5400390625 0.4697265625 0.462890625


================================================
FILE: TumorDetection/train/labels/glioma_358_jpg.rf.8cd4754751b64828d47510309613f1a0.txt
================================================
1 0.603515625 0.6806640625 0.60546875 0.6474609375 0.5947265625 0.6328125 0.5771484375 0.640625 0.568359375 0.6298828125 0.576171875 0.6142578125 0.5654296875 0.59375 0.5419921875 0.595703125 0.5302734375 0.6171875 0.521484375 0.6123046875 0.5341796875 0.59375 0.4931640625 0.59765625 0.47265625 0.6220703125 0.470703125 0.6435546875 0.5 0.6689453125 0.52734375 0.7236328125 0.5390625 0.7255859375 0.552734375 0.7158203125 0.5595703125 0.693359375 0.5888671875 0.69140625 0.603515625 0.6806640625


================================================
FILE: TumorDetection/train/labels/glioma_374_jpg.rf.cd5b6cddcb663677f866adeafb8448a5.txt
================================================
1 0.5078125 0.4501953125 0.4912109375 0.41796875 0.4755859375 0.42578125 0.4580078125 0.412109375 0.4345703125 0.41015625 0.404296875 0.4443359375 0.41015625 0.4873046875 0.404296875 0.5419921875 0.41796875 0.5517578125 0.40625 0.5927734375 0.416015625 0.5947265625 0.42578125 0.5732421875 0.4921875 0.5087890625 0.5078125 0.4697265625 0.5078125 0.4501953125


================================================
FILE: TumorDetection/train/labels/glioma_392_jpg.rf.b7aa80b4d570d52b3122bbafb7961b86.txt
================================================
1 0.375 0.4931640625 0.3291015625 0.46875 0.2939453125 0.4765625 0.267578125 0.5009765625 0.2734375 0.5595703125 0.2900390625 0.572265625 0.3125 0.5732421875 0.3173828125 0.564453125 0.3681640625 0.5546875 0.38671875 0.5263671875 0.375 0.4931640625


================================================
FILE: TumorDetection/train/labels/glioma_414_jpg.rf.f9888de8bd4c5d90586fb849d7d933ed.txt
================================================
1 0.5146484375 0.369140625 0.4970703125 0.3671875 0.48828125 0.3603515625 0.5 0.3544921875 0.4892578125 0.353515625 0.4833984375 0.3671875 0.46875 0.3642578125 0.4833984375 0.3828125 0.50390625 0.3857421875 0.5068359375 0.375 0.5234375 0.3740234375 0.5185546875 0.36328125 0.5146484375 0.369140625
1 0.4931640625 0.25390625 0.46875 0.3369140625 0.466796875 0.3583984375 0.47265625 0.3603515625 0.4921875 0.3232421875 0.517578125 0.3076171875 0.505859375 0.2607421875 0.4931640625 0.25390625


================================================
FILE: TumorDetection/train/labels/glioma_468_jpg.rf.d85e769874f88b31113296c64243faa6.txt
================================================
1 0.6650390625 0.515625 0.6201171875 0.4921875 0.5830078125 0.48828125 0.54296875 0.5166015625 0.533203125 0.5400390625 0.53515625 0.5712890625 0.541015625 0.5869140625 0.5849609375 0.626953125 0.625 0.6298828125 0.6611328125 0.615234375 0.685546875 0.5927734375 0.689453125 0.5615234375 0.6650390625 0.515625


================================================
FILE: TumorDetection/train/labels/glioma_474_jpg.rf.9a407f6530fa06506f37d610565362e1.txt
================================================
1 0.5751953125 0.1796875 0.55859375 0.1904296875 0.560546875 0.2119140625 0.58984375 0.2314453125 0.623046875 0.2060546875 0.623046875 0.1865234375 0.6123046875 0.17578125 0.5751953125 0.1796875


================================================
FILE: TumorDetection/train/labels/glioma_511_jpg.rf.f84ee5cf0c0f3fc5f6af758d55fdfd58.txt
================================================
1 0.31640625 0.3486328125 0.31640625 0.3818359375 0.3310546875 0.40625 0.345703125 0.4091796875 0.36328125 0.3896484375 0.369140625 0.3486328125 0.3408203125 0.33203125 0.31640625 0.3486328125


================================================
FILE: TumorDetection/train/labels/glioma_51_jpg.rf.d45bc240398f7a69aa84fc12190fcffc.txt
================================================
1 0.412109375 0.3583984375 0.43359375 0.3134765625 0.4189453125 0.298828125 0.3955078125 0.296875 0.369140625 0.3193359375 0.3681640625 0.328125 0.349609375 0.3369140625 0.357421875 0.3623046875 0.341796875 0.3818359375 0.4091796875 0.416015625 0.416015625 0.4599609375 0.4365234375 0.4609375 0.44140625 0.4716796875 0.44921875 0.4619140625 0.42578125 0.4033203125 0.42578125 0.3740234375 0.412109375 0.3583984375


================================================
FILE: TumorDetection/train/labels/glioma_522_jpg.rf.f5d4d387b72a07daeecf462fe1f4a151.txt
================================================
1 0.5205078125 0.56640625 0.505859375 0.5615234375 0.5234375 0.5517578125 0.51953125 0.5322265625 0.5087890625 0.521484375 0.48828125 0.5361328125 0.501953125 0.5693359375 0.484375 0.6064453125 0.494140625 0.6552734375 0.490234375 0.7060546875 0.501953125 0.7080078125 0.53125 0.6279296875 0.53515625 0.5751953125 0.5302734375 0.560546875 0.5205078125 0.56640625


================================================
FILE: TumorDetection/train/labels/glioma_525_jpg.rf.5fc207ce06a7678e3ddffb00c411105a.txt
================================================
1 0.4326171875 0.4375 0.416015625 0.4599609375 0.4140625 0.4853515625 0.4267578125 0.521484375 0.4609375 0.5224609375 0.43359375 0.5380859375 0.455078125 0.5439453125 0.462890625 0.5380859375 0.48828125 0.4619140625 0.4697265625 0.4375 0.4326171875 0.4375


================================================
FILE: TumorDetection/train/labels/glioma_543_jpg.rf.5e7686edfc83b46e7eddc407a8649497.txt
================================================
1 0.462890625 0.4970703125 0.462890625 0.5185546875 0.4375 0.5810546875 0.443359375 0.5908203125 0.45703125 0.5771484375 0.486328125 0.5068359375 0.4814453125 0.470703125 0.462890625 0.4970703125
1 0.54296875 0.5126953125 0.54296875 0.4814453125 0.5244140625 0.478515625 0.521484375 0.5087890625 0.556640625 0.5595703125 0.56640625 0.5615234375 0.5830078125 0.552734375 0.5908203125 0.556640625 0.58984375 0.5458984375 0.54296875 0.5126953125


================================================
FILE: TumorDetection/train/labels/glioma_550_jpg.rf.b52127c0a77a8785f44d94766365a326.txt
================================================
1 0.6298828125 0.529296875 0.6044921875 0.541015625 0.6015625 0.5654296875 0.61328125 0.5830078125 0.6357421875 0.603515625 0.66015625 0.6083984375 0.7041015625 0.59765625 0.712890625 0.5732421875 0.6650390625 0.53515625 0.6298828125 0.529296875


================================================
FILE: TumorDetection/train/labels/glioma_551_jpg.rf.aac527f5cb1dcfdfa00a5486b61b973b.txt
================================================
1 0.7578125 0.4931640625 0.759765625 0.4755859375 0.7548828125 0.466796875 0.7470703125 0.474609375 0.7255859375 0.447265625 0.6806640625 0.451171875 0.662109375 0.4736328125 0.658203125 0.5048828125 0.68359375 0.5146484375 0.66796875 0.5205078125 0.6767578125 0.541015625 0.6806640625 0.529296875 0.6904296875 0.53125 0.68359375 0.5400390625 0.7001953125 0.556640625 0.720703125 0.5576171875 0.73828125 0.5498046875 0.73046875 0.5166015625 0.7373046875 0.5 0.7578125 0.4931640625


================================================
FILE: TumorDetection/train/labels/glioma_578_jpg.rf.54b68964d2a5408bbd1bba9919a4488a.txt
================================================
1 0.529296875 0.4697265625 0.5380859375 0.48828125 0.5625 0.4912109375 0.58984375 0.4794921875 0.5830078125 0.458984375 0.5615234375 0.455078125 0.529296875 0.4697265625


================================================
FILE: TumorDetection/train/labels/glioma_609_jpg.rf.e564f70d7e6adff47de4b52b92e46753.txt
================================================
1 0.5419921875 0.26953125 0.5146484375 0.271484375 0.49609375 0.2822265625 0.501953125 0.3369140625 0.5126953125 0.35546875 0.5341796875 0.3671875 0.5615234375 0.361328125 0.611328125 0.3818359375 0.62890625 0.3642578125 0.642578125 0.3271484375 0.6142578125 0.283203125 0.5419921875 0.26953125


================================================
FILE: TumorDetection/train/labels/glioma_629_jpg.rf.563773926f1e45d0ef9ac57c3cfad830.txt
================================================
1 0.4091796875 0.2734375 0.3896484375 0.28515625 0.373046875 0.3037109375 0.36328125 0.3330078125 0.365234375 0.3896484375 0.3779296875 0.3984375 0.396484375 0.3974609375 0.451171875 0.3564453125 0.44921875 0.2900390625 0.4287109375 0.2734375 0.4091796875 0.2734375


================================================
FILE: TumorDetection/train/labels/glioma_650_jpg.rf.10b1beda8b03d28b0450af5be9c0b215.txt
================================================
1 0.5 0.5810546875 0.5166015625 0.578125 0.5146484375 0.5703125 0.5244140625 0.578125 0.5283203125 0.572265625 0.5390625 0.5771484375 0.55078125 0.5263671875 0.5048828125 0.48046875 0.4736328125 0.478515625 0.451171875 0.4931640625 0.443359375 0.5146484375 0.46484375 0.5380859375 0.4423828125 0.537109375 0.44140625 0.5478515625 0.4521484375 0.5625 0.4736328125 0.568359375 0.4775390625 0.578125 0.4912109375 0.578125 0.490234375 0.5908203125 0.5068359375 0.591796875 0.5107421875 0.607421875 0.51953125 0.6064453125 0.525390625 0.6025390625 0.51171875 0.5986328125 0.51953125 0.5849609375 0.5 0.5810546875


================================================
FILE: TumorDetection/train/labels/glioma_665_jpg.rf.c8f631906a0210b39ce82d1f6ee531d9.txt
================================================
1 0.6484375 0.3310546875 0.6337890625 0.30859375 0.6181640625 0.302734375 0.6005859375 0.30859375 0.5888671875 0.30078125 0.5439453125 0.30078125 0.5322265625 0.31640625 0.521484375 0.3173828125 0.513671875 0.3330078125 0.505859375 0.3916015625 0.5341796875 0.412109375 0.580078125 0.4130859375 0.6103515625 0.40625 0.638671875 0.3818359375 0.6484375 0.3583984375 0.6484375 0.3310546875


================================================
FILE: TumorDetection/train/labels/glioma_666_jpg.rf.6e859f6183e409c261f3e872632c0fce.txt
================================================
1 0.58203125 0.3349609375 0.58984375 0.3017578125 0.5576171875 0.298828125 0.5341796875 0.30859375 0.5283203125 0.298828125 0.47014508906249997 0.244140625 0.491071428125 0.30970982187499996 0.486328125 0.3603515625 0.490234375 0.3740234375 0.5 0.3759765625 0.5244140625 0.353515625 0.5556640625 0.359375 0.5673828125 0.3515625 0.58203125 0.3349609375


================================================
FILE: TumorDetection/train/labels/glioma_70_jpg.rf.c053d9e585343a2d87980ac48b2b827a.txt
================================================
1 0.470703125 0.4775390625 0.47265625 0.4697265625 0.4619140625 0.458984375 0.380859375 0.4990234375 0.3828125 0.5556640625 0.37109375 0.5869140625 0.38671875 0.5947265625 0.4140625 0.5419921875 0.408203125 0.5322265625 0.4345703125 0.521484375 0.470703125 0.4775390625
1 0.607421875 0.5791015625 0.591796875 0.5537109375 0.58984375 0.5185546875 0.5400390625 0.466796875 0.52734375 0.4892578125 0.529296875 0.5068359375 0.5625 0.5673828125 0.564453125 0.6025390625 0.580078125 0.6318359375 0.580078125 0.6474609375 0.595703125 0.6552734375 0.6015625 0.6396484375 0.5859375 0.6220703125 0.587890625 0.5908203125 0.607421875 0.5791015625


================================================
FILE: TumorDetection/train/labels/glioma_722_jpg.rf.0973288dd768103902dbadadb02ff063.txt
================================================
1 0.5185546875 0.51171875 0.4599609375 0.5234375 0.44921875 0.5341796875 0.44140625 0.5615234375 0.443359375 0.5810546875 0.470703125 0.6220703125 0.4833984375 0.6328125 0.5 0.6318359375 0.533203125 0.6064453125 0.55859375 0.5576171875 0.5546875 0.5361328125 0.5185546875 0.51171875


================================================
FILE: TumorDetection/train/labels/glioma_835_jpg.rf.7c101a77efb377960f7739ae8d47f4db.txt
================================================
1 0.4033203125 0.44140625 0.4111328125 0.453125 0.3837890625 0.455078125 0.38671875 0.4658203125 0.369140625 0.4775390625 0.3740234375 0.484375 0.3798828125 0.474609375 0.3876953125 0.486328125 0.3974609375 0.4765625 0.3955078125 0.482421875 0.408203125 0.4853515625 0.3984375 0.4970703125 0.408203125 0.4990234375 0.435546875 0.4658203125 0.4375 0.4482421875 0.4287109375 0.435546875 0.4033203125 0.44140625
1 0.349609375 0.5126953125 0.3466796875 0.5390625 0.34375 0.4892578125 0.322265625 0.4873046875 0.39453125 0.4306640625 0.38671875 0.4267578125 0.380859375 0.4013671875 0.3662109375 0.388671875 0.3134765625 0.384765625 0.298828125 0.3994140625 0.296875 0.4404296875 0.27734375 0.4677734375 0.2880859375 0.484375 0.30859375 0.4931640625 0.3037109375 0.5234375 0.3125 0.5224609375 0.3125 0.4970703125 0.32421875 0.4912109375 0.31640625 0.5048828125 0.3203125 0.5185546875 0.30859375 0.5283203125 0.318359375 0.5244140625 0.318359375 0.5400390625 0.3037109375 0.537109375 0.302734375 0.5458984375 0.3095703125 0.552734375 0.353515625 0.5537109375 0.365234375 0.5498046875 0.35546875 0.5341796875 0.357421875 0.5126953125 0.349609375 0.5126953125


================================================
FILE: TumorDetection/train/labels/glioma_904_jpg.rf.46fbdba7d5697ea773c86f0214380230.txt
================================================
1 0.513671875 0.4970703125 0.509765625 0.4560546875 0.4912109375 0.439453125 0.4736328125 0.443359375 0.4765625 0.4365234375 0.4619140625 0.4296875 0.4052734375 0.4296875 0.392578125 0.4462890625 0.39453125 0.4638671875 0.4150390625 0.484375 0.4404296875 0.48828125 0.44140625 0.5244140625 0.46484375 0.5322265625 0.4765625 0.5244140625 0.4775390625 0.5 0.4951171875 0.51171875 0.513671875 0.4970703125


================================================
FILE: TumorDetection/train/labels/glioma_906_jpg.rf.3b56fc6141078c119069f74ca103cd22.txt
================================================
1 0.5205078125 0.427734375 0.4677734375 0.40234375 0.4462890625 0.40234375 0.4091796875 0.41796875 0.39453125 0.4345703125 0.392578125 0.4736328125 0.423828125 0.5029296875 0.41796875 0.5048828125 0.423828125 0.5166015625 0.41796875 0.5283203125 0.427734375 0.5400390625 0.416015625 0.5517578125 0.455078125 0.5595703125 0.517578125 0.5146484375 0.5390625 0.4912109375 0.54296875 0.4697265625 0.5205078125 0.427734375


================================================
FILE: TumorDetection/train/labels/glioma_930_jpg.rf.eb6881f20608f5482378962913095da7.txt
================================================
1 0.5791015625 0.3515625 0.5595703125 0.3515625 0.55078125 0.3603515625 0.5546875 0.3994140625 0.548828125 0.4287109375 0.5751953125 0.458984375 0.6015625 0.4599609375 0.611328125 0.4462890625 0.595703125 0.4228515625 0.59765625 0.3701171875 0.5791015625 0.3515625


================================================
FILE: TumorDetection/train/labels/glioma_933_jpg.rf.6691f90d0b86b8a81e91b978f4d02543.txt
================================================
1 0.45703125 0.4091796875 0.462890625 0.4052734375 0.4580078125 0.3984375 0.427734375 0.3896484375 0.45703125 0.4091796875
1 0.333984375 0.3955078125 0.34375 0.3876953125 0.3359375 0.3857421875 0.3349609375 0.375 0.2939453125 0.365234375 0.27734375 0.3740234375 0.2900390625 0.376953125 0.29296875 0.3857421875 0.2880859375 0.37890625 0.2783203125 0.38671875 0.2734375 0.4052734375 0.294921875 0.4248046875 0.3046875 0.4228515625 0.3056640625 0.408203125 0.3359375 0.4052734375 0.333984375 0.3955078125


================================================
FILE: TumorDetection/train/labels/glioma_97_jpg.rf.1467ac8594412f669ecab1c802dd5b11.txt
================================================
1 0.6943359375 0.634765625 0.677734375 0.6474609375 0.669921875 0.6826171875 0.6826171875 0.69921875 0.69921875 0.7021484375 0.71875 0.6806640625 0.720703125 0.6591796875 0.7060546875 0.63671875 0.6943359375 0.634765625


================================================
FILE: TumorDetection/train/labels/meningioma_1003_jpg.rf.1cb046de3a4bc224cd1d63fb5fa5561f.txt
================================================
2 0.4501953125 0.138671875 0.3994140625 0.1484375 0.38671875 0.1650390625 0.390625 0.2021484375 0.4228515625 0.23046875 0.455078125 0.2333984375 0.48828125 0.2197265625 0.494140625 0.1728515625 0.486328125 0.1572265625 0.4501953125 0.138671875


================================================
FILE: TumorDetection/train/labels/meningioma_101_jpg.rf.bbe32c57f14ff5362085ea9560be430b.txt
================================================
2 0.3740234375 0.529296875 0.353515625 0.5400390625 0.33984375 0.5693359375 0.3583984375 0.61328125 0.40234375 0.6201171875 0.42578125 0.6083984375 0.42578125 0.5732421875 0.3916015625 0.53125 0.3740234375 0.529296875


================================================
FILE: TumorDetection/train/labels/meningioma_1020_jpg.rf.c083f0e6892b08d794d09d0ad935bdd5.txt
================================================
2 0.6533203125 0.404296875 0.6181640625 0.400390625 0.6025390625 0.38671875 0.5830078125 0.38671875 0.564453125 0.4013671875 0.55078125 0.4599609375 0.56640625 0.4912109375 0.58984375 0.5029296875 0.6162109375 0.484375 0.669921875 0.4658203125 0.671875 0.4365234375 0.6533203125 0.404296875


================================================
FILE: TumorDetection/train/labels/meningioma_1030_jpg.rf.7a3c8c1da556e3207c3779e303396f0d.txt
================================================
2 0.6142578125 0.52734375 0.5869140625 0.51953125 0.560546875 0.5322265625 0.52734375 0.5849609375 0.513671875 0.6201171875 0.513671875 0.6455078125 0.5263671875 0.6640625 0.5576171875 0.68359375 0.607421875 0.6845703125 0.6640625 0.6630859375 0.671875 0.5986328125 0.6640625 0.5771484375 0.6142578125 0.52734375


================================================
FILE: TumorDetection/train/labels/meningioma_1031_jpg.rf.92d5d9e4f499a8616701629db1bbf8e6.txt
================================================
2 0.4560546875 0.30078125 0.3857421875 0.28515625 0.3603515625 0.291015625 0.3388671875 0.306640625 0.330078125 0.3271484375 0.353515625 0.4111328125 0.3740234375 0.4296875 0.396484375 0.4287109375 0.447265625 0.3935546875 0.46484375 0.3603515625 0.46875 0.3232421875 0.4560546875 0.30078125


================================================
FILE: TumorDetection/train/labels/meningioma_1032_jpg.rf.6f42b84ac1d4bc6443068a2d2dfe4b84.txt
================================================
2 0.4384765625 0.412109375 0.466796875 0.3759765625 0.4765625 0.3447265625 0.4765625 0.2802734375 0.4560546875 0.263671875 0.3759765625 0.2734375 0.337890625 0.3037109375 0.326171875 0.3310546875 0.3203125 0.3857421875 0.3623046875 0.41796875 0.4130859375 0.423828125 0.4384765625 0.412109375


================================================
FILE: TumorDetection/train/labels/meningioma_1040_jpg.rf.d34a9c9b9fecc69c30ac22cda6a360d0.txt
================================================
2 0.2880859375 0.47265625 0.2666015625 0.478515625 0.248046875 0.5009765625 0.25 0.5224609375 0.2666015625 0.537109375 0.28515625 0.5400390625 0.306640625 0.5283203125 0.314453125 0.5009765625 0.32421875 0.4931640625 0.2880859375 0.47265625


================================================
FILE: TumorDetection/train/labels/meningioma_1048_jpg.rf.a4b0c3d62dc06cc46a19fa7457356dbf.txt
================================================
2 0.5908203125 0.46484375 0.556640625 0.5029296875 0.5546875 0.5419921875 0.5712890625 0.560546875 0.5947265625 0.572265625 0.61328125 0.5712890625 0.671875 0.5205078125 0.666015625 0.4853515625 0.6396484375 0.462890625 0.5908203125 0.46484375


================================================
FILE: TumorDetection/train/labels/meningioma_1051_jpg.rf.a2b35bd71479146c9668177cf6dfa968.txt
================================================
2 0.6787109375 0.2578125 0.66015625 0.2724609375 0.654296875 0.2958984375 0.6708984375 0.31640625 0.693359375 0.3193359375 0.720703125 0.2998046875 0.724609375 0.2841796875 0.7060546875 0.263671875 0.6787109375 0.2578125


================================================
FILE: TumorDetection/train/labels/meningioma_1052_jpg.rf.14993d2fea569cfcd8576a5561bba274.txt
================================================
2 0.697265625 0.3955078125 0.6611328125 0.376953125 0.5810546875 0.375 0.568359375 0.3994140625 0.5703125 0.4169921875 0.5947265625 0.45703125 0.6162109375 0.47265625 0.6328125 0.4716796875 0.68359375 0.4443359375 0.685546875 0.4091796875 0.697265625 0.4033203125 0.697265625 0.3955078125


================================================
FILE: TumorDetection/train/labels/meningioma_1053_jpg.rf.4f69cbaaa68a46b0b815f148e856f8fb.txt
================================================
2 0.5986328125 0.34765625 0.580078125 0.3310546875 0.5830078125 0.322265625 0.5458984375 0.3125 0.5224609375 0.318359375 0.490234375 0.3427734375 0.490234375 0.3662109375 0.5048828125 0.390625 0.5205078125 0.396484375 0.5478515625 0.390625 0.544921875 0.4013671875 0.5693359375 0.42578125 0.580078125 0.4248046875 0.5732421875 0.435546875 0.62109375 0.4384765625 0.650390625 0.4189453125 0.654296875 0.3798828125 0.6455078125 0.359375 0.5986328125 0.34765625


================================================
FILE: TumorDetection/train/labels/meningioma_1067_jpg.rf.2876f08236a9221433638d0e01ceef85.txt
================================================
2 0.701171875 0.3369140625 0.666015625 0.3017578125 0.6767578125 0.30078125 0.67578125 0.2939453125 0.6357421875 0.271484375 0.5576171875 0.275390625 0.537109375 0.2919921875 0.498046875 0.3583984375 0.5 0.3935546875 0.5595703125 0.443359375 0.580078125 0.4462890625 0.6103515625 0.44140625 0.6787109375 0.396484375 0.705078125 0.3525390625 0.701171875 0.3369140625


================================================
FILE: TumorDetection/train/labels/meningioma_1072_jpg.rf.a7c68a0d0357e08de8b940f3c6c69ed4.txt
================================================
2 0.4775390625 0.48046875 0.5068359375 0.484375 0.537109375 0.4599609375 0.544921875 0.4443359375 0.541015625 0.3857421875 0.501953125 0.3427734375 0.490234375 0.2900390625 0.4619140625 0.267578125 0.4365234375 0.267578125 0.4189453125 0.28125 0.3984375 0.3115234375 0.365234375 0.3408203125 0.3515625 0.3681640625 0.35546875 0.4267578125 0.3955078125 0.45703125 0.4423828125 0.462890625 0.4775390625 0.48046875


================================================
FILE: TumorDetection/train/labels/meningioma_1076_jpg.rf.4fa57284674e5fbf07d6446ff67d0a76.txt
================================================
2 0.31640625 0.6083984375 0.314453125 0.6357421875 0.3291015625 0.66015625 0.353515625 0.6708984375 0.375 0.6455078125 0.38671875 0.5947265625 0.3388671875 0.583984375 0.31640625 0.6083984375


================================================
FILE: TumorDetection/train/labels/meningioma_1079_jpg.rf.802709b4226820338bef07534cabe5b6.txt
================================================
2 0.4521484375 0.296875 0.4833984375 0.29296875 0.5029296875 0.28515625 0.521484375 0.2607421875 0.5234375 0.2177734375 0.4931640625 0.185546875 0.4267578125 0.1796875 0.38671875 0.2041015625 0.3828125 0.2314453125 0.392578125 0.2626953125 0.4169921875 0.287109375 0.4521484375 0.296875


================================================
FILE: TumorDetection/train/labels/meningioma_1083_jpg.rf.2f6bc4256771a58ea2c38f4575492a37.txt
================================================
2 0.6884765625 0.30859375 0.69140625 0.2998046875 0.6845703125 0.302734375 0.6806640625 0.291015625 0.6787109375 0.30078125 0.66796875 0.2958984375 0.68359375 0.2841796875 0.689453125 0.2587890625 0.6396484375 0.193359375 0.6005859375 0.189453125 0.5791015625 0.208984375 0.5615234375 0.193359375 0.52734375 0.2294921875 0.529296875 0.2705078125 0.5556640625 0.31640625 0.5849609375 0.333984375 0.609375 0.3330078125 0.6474609375 0.31640625 0.6611328125 0.314453125 0.6650390625 0.32421875 0.6884765625 0.30859375


================================================
FILE: TumorDetection/train/labels/meningioma_1085_jpg.rf.4b6969bbe6585cddc76c5ff75db16419.txt
================================================
2 0.2734375 0.3486328125 0.2470703125 0.32421875 0.2021484375 0.3203125 0.18359375 0.3408203125 0.162109375 0.4326171875 0.1787109375 0.43359375 0.2021484375 0.451171875 0.232421875 0.4541015625 0.2607421875 0.443359375 0.27734375 0.4228515625 0.28515625 0.3740234375 0.2734375 0.3486328125


================================================
FILE: TumorDetection/train/labels/meningioma_1087_jpg.rf.ec345f0d4e2591a385d025fd56355338.txt
================================================
2 0.279296875 0.2958984375 0.2587890625 0.263671875 0.21484375 0.3134765625 0.205078125 0.3603515625 0.2109375 0.3662109375 0.2216796875 0.353515625 0.2587890625 0.349609375 0.27734375 0.3291015625 0.279296875 0.2958984375


================================================
FILE: TumorDetection/train/labels/meningioma_1098_jpg.rf.bad8fc08770434b25e8f030f21da68bc.txt
================================================
2 0.37109375 0.5361328125 0.3603515625 0.51953125 0.3271484375 0.501953125 0.3056640625 0.5 0.2666015625 0.48046875 0.2392578125 0.482421875 0.20703125 0.5166015625 0.201171875 0.5380859375 0.228515625 0.5732421875 0.24609375 0.6220703125 0.2998046875 0.673828125 0.310546875 0.6728515625 0.337890625 0.6572265625 0.376953125 0.6005859375 0.380859375 0.5830078125 0.37109375 0.5361328125
4 0.6474609375 0.36328125 0.5986328125 0.33203125 0.5341796875 0.3125 0.4677734375 0.30859375 0.3994140625 0.330078125 0.375 0.3564453125 0.375 0.3818359375 0.4189453125 0.4453125 0.4365234375 0.451171875 0.4794921875 0.42578125 0.5166015625 0.42578125 0.5537109375 0.439453125 0.5546875 0.4482421875 0.5712890625 0.45703125 0.5810546875 0.45703125 0.5869140625 0.443359375 0.6064453125 0.44921875 0.640625 0.4951171875 0.658203125 0.5009765625 0.6787109375 0.494140625 0.6953125 0.4736328125 0.693359375 0.4228515625 0.6474609375 0.36328125


================================================
FILE: TumorDetection/train/labels/meningioma_1100_jpg.rf.44e77aef5e60423e99363a50dde7c098.txt
================================================
2 0.38671875 0.5673828125 0.3759765625 0.53515625 0.3369140625 0.509765625 0.2998046875 0.50390625 0.2333984375 0.529296875 0.21484375 0.5576171875 0.2265625 0.5712890625 0.232421875 0.6025390625 0.2587890625 0.634765625 0.2822265625 0.650390625 0.3125 0.6513671875 0.3427734375 0.642578125 0.3564453125 0.625 0.3828125 0.6103515625 0.38671875 0.5673828125


================================================
FILE: TumorDetection/train/labels/meningioma_1103_jpg.rf.8f9a1d7d54389a3e56587b3abd7e4a36.txt
================================================
2 0.3271484375 0.568359375 0.3046875 0.5947265625 0.3046875 0.6142578125 0.3271484375 0.640625 0.357421875 0.6435546875 0.39453125 0.6181640625 0.39453125 0.5830078125 0.3662109375 0.564453125 0.3271484375 0.568359375


================================================
FILE: TumorDetection/train/labels/meningioma_1118_jpg.rf.00df375c7507bc402d95140278e74d13.txt
================================================
2 0.455078125 0.4248046875 0.4453125 0.3935546875 0.46875 0.3544921875 0.46875 0.3310546875 0.455078125 0.2763671875 0.4404296875 0.26171875 0.4267578125 0.26171875 0.4091796875 0.275390625 0.3798828125 0.2734375 0.3662109375 0.294921875 0.3408203125 0.28125 0.3173828125 0.28125 0.2734375 0.3154296875 0.267578125 0.3291015625 0.26953125 0.3876953125 0.2900390625 0.408203125 0.31640625 0.4169921875 0.310546875 0.4501953125 0.33203125 0.4658203125 0.4033203125 0.45703125 0.4267578125 0.4609375 0.451171875 0.4423828125 0.455078125 0.4248046875


================================================
FILE: TumorDetection/train/labels/meningioma_1128_jpg.rf.33c271ae0482d8d0c866b985152e5d7c.txt
================================================
2 0.5859375 0.3583984375 0.5810546875 0.34375 0.5615234375 0.33984375 0.53515625 0.3505859375 0.533203125 0.3662109375 0.5478515625 0.3828125 0.5693359375 0.37890625 0.5859375 0.3583984375


================================================
FILE: TumorDetection/train/labels/meningioma_1134_jpg.rf.d1abfc3ea3ca3e76d4889b36dad0d49a.txt
================================================
2 0.3759765625 0.32421875 0.359375 0.3408203125 0.361328125 0.3623046875 0.349609375 0.3701171875 0.341796875 0.3896484375 0.3564453125 0.404296875 0.3876953125 0.40234375 0.427734375 0.4130859375 0.4609375 0.4033203125 0.462890625 0.3603515625 0.4365234375 0.330078125 0.3759765625 0.32421875


================================================
FILE: TumorDetection/train/labels/meningioma_113_jpg.rf.d837df7cfe8a79f90887240e07a400bc.txt
================================================
2 0.5546875 0.4326171875 0.5390625 0.4033203125 0.5263671875 0.39453125 0.4912109375 0.390625 0.4677734375 0.396484375 0.431640625 0.4248046875 0.451171875 0.5166015625 0.484375 0.5361328125 0.5263671875 0.521484375 0.533203125 0.4892578125 0.55859375 0.4697265625 0.5546875 0.4326171875


================================================
FILE: TumorDetection/train/labels/meningioma_1147_jpg.rf.48ec0ae1b9ac50e1614190b9198cfe29.txt
================================================
2 0.798828125 0.2216796875 0.7666015625 0.189453125 0.7158203125 0.16015625 0.685546875 0.1806640625 0.669921875 0.2431640625 0.689453125 0.2939453125 0.7275390625 0.322265625 0.73828125 0.3212890625 0.7998046875 0.31640625 0.83203125 0.2861328125 0.83203125 0.2744140625 0.798828125 0.2216796875


================================================
FILE: TumorDetection/train/labels/meningioma_1153_jpg.rf.cd5615803871925f83c0420197f243ba.txt
================================================
2 0.533203125 0.4853515625 0.5146484375 0.4609375 0.4912109375 0.451171875 0.4208984375 0.46484375 0.404296875 0.4775390625 0.388671875 0.5048828125 0.388671875 0.5458984375 0.40234375 0.5849609375 0.4345703125 0.609375 0.455078125 0.6103515625 0.4951171875 0.591796875 0.51953125 0.5693359375 0.53515625 0.5400390625 0.533203125 0.4853515625


================================================
FILE: TumorDetection/train/labels/meningioma_1166_jpg.rf.57f037b5856750e964733a2977579ea6.txt
================================================
2 0.4404296875 0.40234375 0.45703125 0.3798828125 0.45703125 0.3349609375 0.4482421875 0.318359375 0.4248046875 0.30859375 0.3935546875 0.3125 0.3759765625 0.328125 0.365234375 0.2841796875 0.3251953125 0.26171875 0.2939453125 0.27734375 0.27734375 0.3017578125 0.283203125 0.3134765625 0.259765625 0.3525390625 0.259765625 0.3935546875 0.2958984375 0.4375 0.337890625 0.4443359375 0.375 0.4208984375 0.3837890625 0.396484375 0.4052734375 0.41015625 0.4404296875 0.40234375


================================================
FILE: TumorDetection/train/labels/meningioma_1175_jpg.rf.a4b25fbfaa57544d4497a754fca96349.txt
================================================
2 0.4599609375 0.541015625 0.4794921875 0.533203125 0.5078125 0.5146484375 0.5234375 0.4814453125 0.525390625 0.4501953125 0.513671875 0.4169921875 0.4873046875 0.388671875 0.4541015625 0.373046875 0.3955078125 0.37890625 0.373046875 0.3994140625 0.357421875 0.4306640625 0.36328125 0.4833984375 0.3955078125 0.5234375 0.4228515625 0.537109375 0.4599609375 0.541015625


================================================
FILE: TumorDetection/train/labels/meningioma_1178_jpg.rf.d28ae433a62770aae96bb76711418137.txt
================================================
2 0.5234375 0.3818359375 0.5009765625 0.3359375 0.4033203125 0.328125 0.3828125 0.3427734375 0.365234375 0.3701171875 0.35546875 0.3994140625 0.357421875 0.4287109375 0.373046875 0.4619140625 0.3935546875 0.48046875 0.4169921875 0.4921875 0.44921875 0.4951171875 0.4833984375 0.486328125 0.51171875 0.4580078125 0.5234375 0.4208984375 0.5234375 0.3818359375


================================================
FILE: TumorDetection/train/labels/meningioma_117_jpg.rf.77f8453394784a23e30b31d223402581.txt
================================================
2 0.51953125 0.4677734375 0.529296875 0.4130859375 0.5166015625 0.404296875 0.4755859375 0.408203125 0.455078125 0.4404296875 0.453125 0.4697265625 0.4609375 0.4892578125 0.4990234375 0.48828125 0.51953125 0.4677734375


================================================
FILE: TumorDetection/train/labels/meningioma_1183_jpg.rf.057d898ba6a98d81499ebed48dd2092c.txt
================================================
2 0.5615234375 0.5625 0.5771484375 0.5625 0.5908203125 0.556640625 0.59765625 0.5419921875 0.5615234375 0.494140625 0.5302734375 0.474609375 0.4931640625 0.46875 0.4619140625 0.474609375 0.44140625 0.5087890625 0.419921875 0.5283203125 0.412109375 0.5498046875 0.412109375 0.5888671875 0.4326171875 0.62109375 0.4599609375 0.634765625 0.4736328125 0.6328125 0.4892578125 0.6484375 0.50390625 0.6513671875 0.51953125 0.6337890625 0.53515625 0.5478515625 0.5439453125 0.544921875 0.5615234375 0.5625


================================================
FILE: TumorDetection/train/labels/meningioma_1185_jpg.rf.af26d6157a2ec8fdb58a10c755004c7b.txt
================================================
2 0.66015625 0.4697265625 0.66796875 0.4404296875 0.662109375 0.4169921875 0.5927734375 0.36328125 0.5361328125 0.3359375 0.525390625 0.3642578125 0.51171875 0.3779296875 0.51171875 0.4326171875 0.5576171875 0.474609375 0.6181640625 0.484375 0.66015625 0.4697265625


================================================
FILE: TumorDetection/train/labels/meningioma_1195_jpg.rf.baa094c6182811a33bf00b56230f0076.txt
================================================
2 0.3740234375 0.5390625 0.3056640625 0.552734375 0.2763671875 0.529296875 0.2451171875 0.533203125 0.2265625 0.5576171875 0.224609375 0.5927734375 0.212890625 0.6083984375 0.22265625 0.6513671875 0.2587890625 0.69140625 0.2978515625 0.720703125 0.3369140625 0.736328125 0.361328125 0.7353515625 0.4169921875 0.705078125 0.451171875 0.6572265625 0.435546875 0.5849609375 0.3740234375 0.5390625


================================================
FILE: TumorDetection/train/labels/meningioma_1200_jpg.rf.8f1194582c8142887ff4537a9e41c294.txt
================================================
2 0.3837890625 0.23046875 0.3564453125 0.21875 0.3427734375 0.21875 0.3076171875 0.2421875 0.251953125 0.3017578125 0.2822265625 0.35546875 0.3095703125 0.373046875 0.345703125 0.3740234375 0.3759765625 0.365234375 0.404296875 0.3173828125 0.40234375 0.2529296875 0.3837890625 0.23046875


================================================
FILE: TumorDetection/train/labels/meningioma_1201_jpg.rf.4155c14224b63c3a4bbf6d570a2a4089.txt
================================================
2 0.3779296875 0.232421875 0.3544921875 0.2265625 0.3154296875 0.23828125 0.25390625 0.3017578125 0.25390625 0.3134765625 0.2734375 0.3271484375 0.279296875 0.3466796875 0.3037109375 0.369140625 0.349609375 0.3740234375 0.3759765625 0.36328125 0.404296875 0.3232421875 0.396484375 0.2490234375 0.3779296875 0.232421875


================================================
FILE: TumorDetection/train/labels/meningioma_120_jpg.rf.f86ba2e7977ba292b26c91f2564d946a.txt
================================================
2 0.6337890625 0.55859375 0.5986328125 0.53515625 0.5517578125 0.544921875 0.53125 0.5751953125 0.537109375 0.6123046875 0.5498046875 0.623046875 0.61328125 0.6279296875 0.658203125 0.6044921875 0.65234375 0.5810546875 0.638671875 0.5732421875 0.6337890625 0.55859375


================================================
FILE: TumorDetection/train/labels/meningioma_1210_jpg.rf.9ae7dc5b727491b6603d19416676a7c8.txt
================================================
2 0.337890625 0.5224609375 0.3154296875 0.501953125 0.2236328125 0.46875 0.2041015625 0.470703125 0.1650390625 0.49609375 0.16015625 0.5087890625 0.166015625 0.5263671875 0.20703125 0.5634765625 0.228515625 0.6201171875 0.2626953125 0.650390625 0.294921875 0.6494140625 0.328125 0.6201171875 0.32421875 0.5849609375 0.341796875 0.5537109375 0.337890625 0.5224609375


================================================
FILE: TumorDetection/train/labels/meningioma_1212_jpg.rf.72220ada901dcb13a50fcfd9483ee956.txt
================================================
2 0.66015625 0.7607421875 0.654296875 0.7255859375 0.6142578125 0.69140625 0.5693359375 0.693359375 0.52734375 0.7412109375 0.533203125 0.7783203125 0.525390625 0.8017578125 0.53515625 0.8115234375 0.537109375 0.8291015625 0.5634765625 0.849609375 0.580078125 0.8486328125 0.611328125 0.8310546875 0.609375 0.8056640625 0.6435546875 0.787109375 0.66015625 0.7607421875


================================================
FILE: TumorDetection/train/labels/meningioma_1220_jpg.rf.163bb25095405d38a11f58b0fb559c0b.txt
================================================
2 0.6484375 0.7197265625 0.6298828125 0.701171875 0.6123046875 0.6953125 0.5810546875 0.69140625 0.5615234375 0.69921875 0.529296875 0.7392578125 0.529296875 0.7626953125 0.5751953125 0.798828125 0.623046875 0.7978515625 0.6396484375 0.791015625 0.65625 0.7705078125 0.658203125 0.7412109375 0.6484375 0.7197265625


================================================
FILE: TumorDetection/train/labels/meningioma_1233_jpg.rf.d62273e91257dbcb25bdaac2b1cd42af.txt
================================================
2 0.4140625 0.3798828125 0.404296875 0.3427734375 0.3642578125 0.30078125 0.3310546875 0.2890625 0.2919921875 0.287109375 0.2724609375 0.294921875 0.251953125 0.3154296875 0.232421875 0.3583984375 0.232421875 0.3779296875 0.26171875 0.4619140625 0.2900390625 0.48046875 0.31640625 0.4794921875 0.3603515625 0.458984375 0.3740234375 0.4296875 0.408203125 0.4150390625 0.4140625 0.3798828125


================================================
FILE: TumorDetection/train/labels/meningioma_1234_jpg.rf.0c1c2a6eb11bffeb99e313f5685f9b87.txt
================================================
2 0.2783203125 0.48046875 0.3154296875 0.482421875 0.3388671875 0.474609375 0.3896484375 0.421875 0.4140625 0.4072265625 0.416015625 0.3623046875 0.40625 0.3369140625 0.3740234375 0.30078125 0.3505859375 0.287109375 0.3115234375 0.283203125 0.2822265625 0.2890625 0.2490234375 0.30859375 0.228515625 0.3408203125 0.224609375 0.3818359375 0.23828125 0.4052734375 0.244140625 0.4501953125 0.2783203125 0.48046875


================================================
FILE: TumorDetection/train/labels/meningioma_1235_jpg.rf.6d4ea4c1965d6fbaeac495438bc29a62.txt
================================================
2 0.43359375 0.4560546875 0.439453125 0.4267578125 0.404296875 0.3447265625 0.3720703125 0.3125 0.3427734375 0.29296875 0.2861328125 0.291015625 0.263671875 0.3076171875 0.24609375 0.3564453125 0.240234375 0.4013671875 0.2578125 0.4384765625 0.30078125 0.4794921875 0.302734375 0.5029296875 0.3330078125 0.52734375 0.361328125 0.5283203125 0.3984375 0.5146484375 0.404296875 0.4892578125 0.421875 0.4794921875 0.43359375 0.4560546875


================================================
FILE: TumorDetection/train/labels/meningioma_1242_jpg.rf.ddf04a2d94430995ea1db196c804f82f.txt
================================================
2 0.494140625 0.5224609375 0.5205078125 0.54296875 0.5478515625 0.546875 0.5810546875 0.53515625 0.6015625 0.5205078125 0.6259765625 0.482421875 0.662109375 0.4619140625 0.66796875 0.4228515625 0.658203125 0.4013671875 0.64453125 0.3779296875 0.6005859375 0.34375 0.4873046875 0.310546875 0.4765625 0.3115234375 0.47265625 0.3310546875 0.46875 0.4072265625 0.478515625 0.5009765625 0.4873046875 0.501953125 0.494140625 0.5224609375


================================================
FILE: TumorDetection/train/labels/meningioma_1243_jpg.rf.0024eff03061b426a5f4a93cc1b539cd.txt
================================================
2 0.65234375 0.4404296875 0.654296875 0.3974609375 0.6138392859375 0.3766741078125 0.6012834828125 0.37248883906249997 0.5873325890625 0.3655133921875 0.5185546875 0.310546875 0.4951171875 0.310546875 0.482421875 0.3193359375 0.474609375 0.3935546875 0.486328125 0.5146484375 0.5263671875 0.529296875 0.564453125 0.5283203125 0.587890625 0.5107421875 0.5927734375 0.49609375 0.63671875 0.4658203125 0.65234375 0.4404296875


================================================
FILE: TumorDetection/train/labels/meningioma_1249_jpg.rf.5ae9d159cbf1096b85e4ab138757395f.txt
================================================
2 0.359375 0.4189453125 0.3515625 0.4013671875 0.3388671875 0.390625 0.3173828125 0.390625 0.2607421875 0.326171875 0.2236328125 0.3359375 0.205078125 0.3759765625 0.150390625 0.4228515625 0.169921875 0.4326171875 0.162109375 0.4833984375 0.177734375 0.5244140625 0.2177734375 0.548828125 0.2314453125 0.578125 0.25390625 0.5830078125 0.3095703125 0.5703125 0.3203125 0.5595703125 0.359375 0.4716796875 0.359375 0.4189453125


================================================
FILE: TumorDetection/train/labels/meningioma_1250_jpg.rf.a13f606fa448cebff1712b9bf916612d.txt
================================================
2 0.236328125 0.6220703125 0.224609375 0.6123046875 0.234375 0.6044921875 0.220703125 0.5986328125 0.23828125 0.5888671875 0.232421875 0.5830078125 0.236328125 0.5751953125 0.2265625 0.5712890625 0.2294921875 0.5625 0.2548828125 0.556640625 0.2998046875 0.572265625 0.3125 0.5595703125 0.326171875 0.5244140625 0.353515625 0.5009765625 0.36328125 0.4794921875 0.35546875 0.4462890625 0.36328125 0.4189453125 0.287109375 0.3720703125 0.28515625 0.3603515625 0.302734375 0.3544921875 0.294921875 0.3505859375 0.2998046875 0.337890625 0.2744140625 0.337890625 0.2021484375 0.365234375 0.16796875 0.4052734375 0.146484375 0.4599609375 0.146484375 0.5107421875 0.166015625 0.5556640625 0.2255859375 0.626953125 0.236328125 0.6220703125


================================================
FILE: TumorDetection/train/labels/meningioma_1255_jpg.rf.f1ab0ba8c39a1ba9c030649227ad01f4.txt
================================================
2 0.51171875 0.3779296875 0.509765625 0.4033203125 0.537109375 0.4306640625 0.5625 0.4072265625 0.564453125 0.3857421875 0.5224609375 0.359375 0.51171875 0.3779296875


================================================
FILE: TumorDetection/train/labels/meningioma_1256_jpg.rf.9b22b2792f72a75a8a7b76642ee9d16b.txt
================================================
2 0.4951171875 0.390625 0.484375 0.4091796875 0.4990234375 0.43359375 0.517578125 0.4384765625 0.541015625 0.4267578125 0.537109375 0.3955078125 0.5224609375 0.3828125 0.4951171875 0.390625


================================================
FILE: TumorDetection/train/labels/meningioma_1269_jpg.rf.ba78521135708aa900973fc31f1144e9.txt
================================================
2 0.5361328125 0.541015625 0.5859375 0.5244140625 0.5859375 0.4931640625 0.609375 0.4814453125 0.611328125 0.4638671875 0.5751953125 0.419921875 0.560546875 0.4228515625 0.5400390625 0.37890625 0.5224609375 0.376953125 0.5048828125 0.38671875 0.4931640625 0.40625 0.4755859375 0.40625 0.46875 0.4306640625 0.466796875 0.4658203125 0.4794921875 0.517578125 0.5029296875 0.53515625 0.5361328125 0.541015625
2 0.5224609375 0.376953125 0.50390625 0.3896484375 0.501953125 0.4111328125 0.5302734375 0.435546875 0.533203125 0.3798828125 0.5224609375 0.376953125


================================================
FILE: TumorDetection/train/labels/meningioma_1280_jpg.rf.1ed02c7f20cb4cbcd709e7e327e36016.txt
================================================
2 0.5771484375 0.61328125 0.544921875 0.6337890625 0.5390625 0.6728515625 0.5546875 0.7294921875 0.595703125 0.7412109375 0.6171875 0.7255859375 0.6328125 0.6865234375 0.607421875 0.6298828125 0.5771484375 0.61328125


================================================
FILE: TumorDetection/train/labels/meningioma_1281_jpg.rf.156dc24e062ef1db758b7923891a7474.txt
================================================
2 0.505859375 0.5185546875 0.505859375 0.4794921875 0.4892578125 0.4609375 0.4560546875 0.451171875 0.42578125 0.4775390625 0.423828125 0.5205078125 0.4462890625 0.537109375 0.4794921875 0.541015625 0.505859375 0.5185546875


================================================
FILE: TumorDetection/train/labels/meningioma_1282_jpg.rf.99fa3552b33841ac55cc812908286092.txt
================================================
2 0.7080078125 0.599609375 0.6455078125 0.60546875 0.64453125 0.6787109375 0.6650390625 0.693359375 0.701171875 0.6982421875 0.7265625 0.6767578125 0.732421875 0.6474609375 0.728515625 0.6162109375 0.7080078125 0.599609375


================================================
FILE: TumorDetection/train/labels/meningioma_1297_jpg.rf.a9a4f7f04148883e328de995b2b18a61.txt
================================================
2 0.62890625 0.6396484375 0.62890625 0.6142578125 0.615234375 0.5908203125 0.5458984375 0.525390625 0.5283203125 0.517578125 0.4755859375 0.51953125 0.4296875 0.5537109375 0.431640625 0.6201171875 0.44140625 0.6650390625 0.455078125 0.6904296875 0.4609375 0.7236328125 0.5 0.7744140625 0.509765625 0.8017578125 0.5361328125 0.814453125 0.552734375 0.8134765625 0.5908203125 0.79296875 0.642578125 0.7255859375 0.642578125 0.6806640625 0.62890625 0.6396484375


================================================
FILE: TumorDetection/train/labels/meningioma_129_jpg.rf.02cf3f9072386c01e700156af5c8f3a1.txt
================================================
2 0.404296875 0.2490234375 0.3662109375 0.22265625 0.3310546875 0.224609375 0.3046875 0.2431640625 0.291015625 0.2841796875 0.267578125 0.3134765625 0.2578125 0.3505859375 0.2578125 0.3603515625 0.2958984375 0.400390625 0.345703125 0.4072265625 0.375 0.3818359375 0.412109375 0.3095703125 0.4140625 0.2861328125 0.404296875 0.2490234375


================================================
FILE: TumorDetection/train/labels/meningioma_1300_jpg.rf.1205de8340edfbe7473552dc47010138.txt
================================================
2 0.62890625 0.6826171875 0.599609375 0.6318359375 0.59375 0.6025390625 0.578125 0.5849609375 0.58203125 0.5595703125 0.5615234375 0.529296875 0.5322265625 0.513671875 0.4580078125 0.521484375 0.447265625 0.5419921875 0.451171875 0.5732421875 0.4296875 0.6142578125 0.44140625 0.6826171875 0.451171875 0.7021484375 0.484375 0.7314453125 0.4912109375 0.75 0.5048828125 0.75 0.5322265625 0.78125 0.552734375 0.7841796875 0.5830078125 0.779296875 0.60546875 0.7666015625 0.62890625 0.7138671875 0.62890625 0.6826171875


================================================
FILE: TumorDetection/train/labels/meningioma_1307_jpg.rf.1562868b20f5f63c07ec0c9eb9121700.txt
================================================
2 0.5380859375 0.458984375 0.4814453125 0.458984375 0.462890625 0.4697265625 0.4765625 0.4990234375 0.501953125 0.5244140625 0.5107421875 0.552734375 0.5419921875 0.564453125 0.578125 0.5654296875 0.59765625 0.5556640625 0.611328125 0.5107421875 0.5693359375 0.46875 0.5380859375 0.458984375


================================================
FILE: TumorDetection/train/labels/meningioma_1318_jpg.rf.c6a8e0e52fae3f2a948ece1d1763f2e9.txt
================================================
2 0.609375 0.7177734375 0.6328125 0.6552734375 0.6259765625 0.61328125 0.5966796875 0.59765625 0.5322265625 0.595703125 0.498046875 0.6181640625 0.494140625 0.6494140625 0.5087890625 0.697265625 0.5712890625 0.7265625 0.609375 0.7177734375


================================================
FILE: TumorDetection/train/labels/meningioma_1319_jpg.rf.51b10a74243da6bcaa41ebce7532006a.txt
================================================
2 0.6162109375 0.708984375 0.6328125 0.6845703125 0.638671875 0.6240234375 0.625 0.5986328125 0.6044921875 0.587890625 0.5439453125 0.595703125 0.5224609375 0.60546875 0.5 0.6318359375 0.513671875 0.7099609375 0.5302734375 0.7265625 0.5771484375 0.728515625 0.6162109375 0.708984375


================================================
FILE: TumorDetection/train/labels/meningioma_1323_jpg.rf.edfca5eda6a7b4f9f60ff05875a85e07.txt
================================================
2 0.771484375 0.5693359375 0.7646484375 0.525390625 0.7470703125 0.5234375 0.7392578125 0.53125 0.7021484375 0.51171875 0.6357421875 0.529296875 0.6142578125 0.54296875 0.59375 0.5693359375 0.5859375 0.5966796875 0.607421875 0.6396484375 0.60546875 0.6572265625 0.5859375 0.6904296875 0.59375 0.7099609375 0.6396484375 0.7265625 0.6953125 0.7333984375 0.7109375 0.7255859375 0.75390625 0.6474609375 0.765625 0.6181640625 0.771484375 0.5693359375


================================================
FILE: TumorDetection/train/labels/meningioma_1336_jpg.rf.7f86928168516cd98a36481db162a727.txt
================================================
2 0.361328125 0.6748046875 0.353515625 0.5986328125 0.36328125 0.5888671875 0.36328125 0.5732421875 0.3447265625 0.556640625 0.3154296875 0.564453125 0.310546875 0.5771484375 0.31640625 0.5947265625 0.2822265625 0.603515625 0.2666015625 0.625 0.2490234375 0.6015625 0.236328125 0.6025390625 0.267578125 0.7021484375 0.2802734375 0.71875 0.287109375 0.7177734375 0.30078125 0.7060546875 0.2978515625 0.701171875 0.3330078125 0.6953125 0.361328125 0.6748046875


================================================
FILE: TumorDetection/train/labels/meningioma_134_jpg.rf.974249c3aefbd2743a3134e9f4340fe8.txt
================================================
2 0.7333984375 0.369140625 0.6845703125 0.375 0.662109375 0.3916015625 0.654296875 0.4111328125 0.658203125 0.4658203125 0.6826171875 0.494140625 0.71484375 0.4990234375 0.7490234375 0.48046875 0.763671875 0.4501953125 0.7578125 0.3857421875 0.7333984375 0.369140625


================================================
FILE: TumorDetection/train/labels/meningioma_136_jpg.rf.8491175983470b61181aad86eb939f8d.txt
================================================
2 0.767578125 0.3876953125 0.712890625 0.3017578125 0.6982421875 0.2890625 0.6728515625 0.28515625 0.6357421875 0.302734375 0.6171875 0.3271484375 0.6171875 0.3583984375 0.626953125 0.3857421875 0.623046875 0.4072265625 0.638671875 0.4462890625 0.6689453125 0.486328125 0.716796875 0.4912109375 0.7451171875 0.48046875 0.759765625 0.4658203125 0.76953125 0.4306640625 0.767578125 0.3876953125


================================================
FILE: TumorDetection/train/labels/meningioma_138_jpg.rf.6b7cc548f3be2b200133aaaf73d5a989.txt
================================================
2 0.7041015625 0.32421875 0.6826171875 0.32421875 0.63671875 0.3486328125 0.625 0.3896484375 0.626953125 0.4345703125 0.6328125 0.4482421875 0.6669921875 0.474609375 0.72265625 0.4794921875 0.7578125 0.4736328125 0.740234375 0.3701171875 0.732421875 0.3505859375 0.7041015625 0.32421875


================================================
FILE: TumorDetection/train/labels/meningioma_139_jpg.rf.dadc43b47996c23ec8abd05dd8a078aa.txt
================================================
2 0.712890625 0.3544921875 0.6865234375 0.34765625 0.6591796875 0.361328125 0.6474609375 0.35546875 0.6435546875 0.38671875 0.63671875 0.3623046875 0.634765625 0.4248046875 0.6630859375 0.45703125 0.697265625 0.4619140625 0.7119140625 0.4609375 0.728515625 0.4404296875 0.72265625 0.3779296875 0.712890625 0.3544921875


================================================
FILE: TumorDetection/train/labels/meningioma_141_jpg.rf.37375ab0537220a06979363c6e8b436f.txt
================================================
2 0.6552734375 0.59765625 0.623046875 0.6162109375 0.61328125 0.6494140625 0.6435546875 0.68359375 0.66796875 0.6845703125 0.6953125 0.6552734375 0.693359375 0.6240234375 0.6826171875 0.60546875 0.6552734375 0.59765625


================================================
FILE: TumorDetection/train/labels/meningioma_147_jpg.rf.1bbf03ae4cebe580fa339f3085bf854b.txt
================================================
2 0.5302734375 0.14453125 0.4853515625 0.1484375 0.466796875 0.1884765625 0.466796875 0.2177734375 0.478515625 0.2392578125 0.5068359375 0.259765625 0.560546875 0.2646484375 0.603515625 0.2314453125 0.61328125 0.2080078125 0.609375 0.1787109375 0.5302734375 0.14453125


================================================
FILE: TumorDetection/train/labels/meningioma_148_jpg.rf.9c6edd3b213eac261b1f461446504426.txt
================================================
2 0.5224609375 0.263671875 0.5615234375 0.265625 0.5849609375 0.255859375 0.603515625 0.2373046875 0.61328125 0.2138671875 0.61328125 0.1943359375 0.599609375 0.1669921875 0.5654296875 0.154296875 0.5478515625 0.138671875 0.5029296875 0.140625 0.46875 0.1728515625 0.466796875 0.2099609375 0.474609375 0.2294921875 0.4931640625 0.25 0.5224609375 0.263671875


================================================
FILE: TumorDetection/train/labels/meningioma_155_jpg.rf.acfb1ac57a47166618309233d3e09e69.txt
================================================
2 0.7109375 0.6142578125 0.6748046875 0.57421875 0.6376953125 0.576171875 0.6259765625 0.56640625 0.5888671875 0.560546875 0.54296875 0.6103515625 0.53515625 0.6943359375 0.5693359375 0.73828125 0.599609375 0.7412109375 0.6630859375 0.7109375 0.708984375 0.6630859375 0.71484375 0.6474609375 0.7109375 0.6142578125


================================================
FILE: TumorDetection/train/labels/meningioma_158_jpg.rf.3b1a3d322fc03483cb5d55d34fcd06fe.txt
================================================
2 0.5791015625 0.7734375 0.6162109375 0.734375 0.6455078125 0.724609375 0.6640625 0.6787109375 0.6123046875 0.626953125 0.5576171875 0.611328125 0.4970703125 0.6328125 0.46484375 0.6650390625 0.4609375 0.7236328125 0.466796875 0.7607421875 0.5478515625 0.7734375 0.5791015625 0.7734375


================================================
FILE: TumorDetection/train/labels/meningioma_161_jpg.rf.508b6ec1ca195f8c4cf6e18d2231c18a.txt
================================================
2 0.4716796875 0.427734375 0.490234375 0.4013671875 0.486328125 0.3759765625 0.4638671875 0.3671875 0.4423828125 0.37890625 0.435546875 0.3857421875 0.4375 0.4072265625 0.4580078125 0.42578125 0.4716796875 0.427734375


================================================
FILE: TumorDetection/train/labels/meningioma_168_jpg.rf.635b9d1dafc64618f2c47d441d871c9c.txt
================================================
2 0.8448046874999999 0.640625 0.7119140640625 0.6132812515625 0.6633984359375 0.6308593734375 0.691875 0.6337890625 0.6486328125 0.6425781265625 0.6053906234374999 0.6845703109375 0.5674218765625 0.8583984375 0.6148828125 0.9375 0.675 0.9482421890625 0.7709765640625 0.8867187484375 0.8859375 0.7587890625 0.9007031234375 0.7080078109375 0.8448046874999999 0.640625


================================================
FILE: TumorDetection/train/labels/meningioma_169_jpg.rf.3496fb138a70dfd25edb4dbb8e71707b.txt
================================================
2 0.7241169562499999 0.736328125 0.79867501875 0.755859375 0.8257870421875 0.751953125 0.8585474046874999 0.7314453125 0.8856594281250001 0.6845703125 0.878881421875 0.5166015625 0.8619364062500001 0.3984375 0.7783410015625 0.396484375 0.7128202796875 0.4257812484375 0.66650390625 0.4658203125 0.6280952078125 0.5361328125 0.6258358703125 0.5830078125 0.655207228125 0.6708984375 0.6970049328125001 0.71875 0.7241169562499999 0.736328125


================================================
FILE: TumorDetection/train/labels/meningioma_174_jpg.rf.c253ac6b03709518b654cf1b846071f9.txt
================================================
2 0.8299198546875 0.568359375 0.8538927796875001 0.5146484359375 0.8630253234374999 0.3115234359375 0.8230704484375 0.2988281234375 0.7659920515625 0.302734375 0.6872238687500001 0.3583984359375 0.6712419171875 0.3916015640625 0.6712419171875 0.4619140625 0.689507003125 0.5166015640625 0.7614257828125 0.5703125015625 0.8299198546875 0.568359375


================================================
FILE: TumorDetection/train/labels/meningioma_185_jpg.rf.b5013a5b232b32fe0e1daa6e17db4db2.txt
================================================
2 0.3985992234375 0.3076171875 0.31954742343750003 0.2578125 0.27723800625 0.263671875 0.2349285921875 0.287109375 0.1781449046875 0.3759765609375 0.18705215 0.4169921875 0.2527430828125 0.474609375 0.27167098125 0.4736328125 0.3373619140625 0.447265625 0.37967132812500004 0.404296875 0.41196009218749996 0.3916015609375 0.3985992234375 0.3076171875


================================================
FILE: TumorDetection/train/labels/meningioma_187_jpg.rf.63e58c7b1736a9c3757b1e9189626f4d.txt
================================================
2 0.7314301125 0.46875 0.677768571875 0.462890625 0.6229404781250001 0.5361328109375 0.6229404781250001 0.6376953125 0.688267571875 0.7509765640625 0.723264225 0.7568359375 0.835253525 0.7119140625 0.84225285625 0.6162109375 0.783925096875 0.5009765640625 0.7314301125 0.46875


================================================
FILE: TumorDetection/train/labels/meningioma_189_jpg.rf.41e99170113b9892ffc42d2040231ad3.txt
================================================
2 0.786162109375 0.3603515609375 0.7632958999999999 0.3398437484375 0.7415185546874999 0.330078125 0.7044970703125 0.3320312515625 0.7034082046875 0.3427734390625 0.712119140625 0.3466796859375 0.7110302749999999 0.3515625015625 0.7001416 0.3496093765625 0.682719725 0.3554687484375 0.6631201156250001 0.373046875 0.6195654296875001 0.3789062515625 0.6021435546875 0.3945312515625 0.591254884375 0.3945312515625 0.583632809375 0.4072265609375 0.5858105453125 0.4287109375 0.5956103500000001 0.439453125 0.621743165625 0.4472656234375 0.660942384375 0.4726562515625 0.699052734375 0.4755859375 0.7741845703125 0.4433593765625 0.7905175796874999 0.4111328109375 0.7905175796874999 0.3720703140625 0.786162109375 0.3603515609375


================================================
FILE: TumorDetection/train/labels/meningioma_190_jpg.rf.906cfd2d2b77ca580dfa533136bfd3d6.txt
================================================
2 0.6154296859375 0.5673828109375 0.5267089828125 0.4941406265625 0.4907714828125 0.50390625 0.4728027328125 0.4902343734375 0.416650390625 0.49609375 0.3234375 0.5869140640625 0.34140625 0.6357421890625 0.3649902328125 0.658203125 0.4626953140625 0.6884765625 0.519970703125 0.685546875 0.5828613265625 0.6621093734375 0.62666015625 0.6005859359375 0.6154296859375 0.5673828109375


================================================
FILE: TumorDetection/train/labels/meningioma_195_jpg.rf.5da508c5656d120630a308e8b43e8f31.txt
================================================
2 0.7109375 0.251620678125 0.7148437484375 0.21089594374999998 0.708984375 0.18762466718749998 0.6533203125 0.1163563828125 0.5927734375 0.11344747343749999 0.5732421875 0.1250831109375 0.5585937484375 0.1527177515625 0.5507812515625 0.1992603046875 0.5351562515625 0.239985040625 0.53125 0.3010721421875 0.552734375 0.3534325125 0.5634765625 0.3665226046875 0.609375 0.3679770609375 0.6298828125 0.35488696875 0.6513671875 0.325797871875 0.6787109375 0.3112533234375 0.7109375 0.251620678125


================================================
FILE: TumorDetection/train/labels/meningioma_196_jpg.rf.0d1f270cff963f8af36775e06432849e.txt
================================================
2 0.44136939999999997 0.3994140625 0.3837522109375 0.337890625 0.29352152187499997 0.3173828140625 0.2956957546875 0.2900390625 0.27069206874999996 0.236328125 0.21633623281249997 0.2148437515625 0.1793742640625 0.2148437515625 0.10218897499999999 0.2919921859375 0.1043632078125 0.3427734375 0.07392393750000001 0.4619140625 0.07827240625000001 0.5029296859375 0.1011018578125 0.5273437515625 0.1511092265625 0.546875 0.23590433281249998 0.5507812484375 0.2728663015625 0.5273437515625 0.2815632359375 0.546875 0.326135025 0.5576171859375 0.3881006796875 0.525390625 0.4218012953125 0.4931640625 0.44789209999999996 0.4345703140625 0.44136939999999997 0.3994140625


================================================
FILE: TumorDetection/train/labels/meningioma_201_jpg.rf.409fa0316610f1968bcde41bd4fdde82.txt
================================================
2 0.8829985125 0.5791015609375 0.8489467062499999 0.5253906234375 0.7808430984375 0.5058593765625 0.74092029375 0.515625 0.6810360875 0.5615234390625 0.659900484375 0.6318359375 0.6963006859374999 0.6835937484375 0.773797896875 0.7265625015625 0.8031529015625001 0.7294921875 0.875953309375 0.6630859375 0.875953309375 0.6435546859375 0.8548177062500001 0.6259765609375 0.8829985125 0.5791015609375


================================================
FILE: TumorDetection/train/labels/meningioma_203_jpg.rf.ea11dc83894c1cd9262ca4d2f1854c40.txt
================================================
2 0.5185546875 0.189453125 0.505859375 0.2060546875 0.5039062484375 0.2490234359375 0.5263671875 0.263671875 0.546875 0.2626953125 0.5703125 0.2314453125 0.568359375 0.1982421875 0.5556640640625 0.189453125 0.5185546875 0.189453125


================================================
FILE: TumorDetection/train/labels/meningioma_206_jpg.rf.9882ab4219a33378e9c675ce3bbec31e.txt
================================================
2 0.8529687499999999 0.4326171875 0.8261458328125 0.3857421875 0.7792057296875 0.3652343734375 0.7336067703125 0.3203125 0.6504557296875 0.30859375 0.5418229171875 0.4306640625 0.5203645828125001 0.4755859375 0.528411459375 0.5205078125 0.5512109359374999 0.5253906265625 0.5364583328125 0.5283203125 0.5699869796875 0.5410156265625 0.5874218734375 0.5654296875 0.6370442703125 0.59765625 0.7362890640625 0.5917968734375 0.7644531265625 0.5986328125 0.803346353125 0.5917968734375 0.8449218734375 0.5615234375 0.8610156265625 0.5068359375 0.8529687499999999 0.4326171875


================================================
FILE: TumorDetection/train/labels/meningioma_222_jpg.rf.bf76ef8786bdbe5ab070a12f63e910ec.txt
================================================
2 0.6992187515625 0.1884765640625 0.6513671875 0.15625 0.5693359359375 0.123046875 0.5283203125 0.119140625 0.5068359359375 0.1289062484375 0.490234375 0.1494140640625 0.505859375 0.2353515640625 0.5732421875 0.287109375 0.642578125 0.2919921875 0.6923828125 0.267578125 0.7109375 0.2431640640625 0.7109375 0.2099609359375 0.6992187515625 0.1884765640625


================================================
FILE: TumorDetection/train/labels/meningioma_223_jpg.rf.7755bf1b15746ddda3a67acee86065a4.txt
================================================
2 0.6142578125 0.314453125 0.662109375 0.2197265640625 0.6904296875 0.21875 0.693359375 0.2119140640625 0.6669921875 0.201171875 0.5595703125 0.1953125 0.5146484359375 0.201171875 0.501953125 0.2177734359375 0.5078125 0.2900390640625 0.5400390640625 0.326171875 0.5693359359375 0.3320312484375 0.6142578125 0.314453125


================================================
FILE: TumorDetection/train/labels/meningioma_225_jpg.rf.9522022bbfd91e2a307e855ac319cf27.txt
================================================
2 0.5302734375 0.2716323390625 0.5771484375 0.2533883765625 0.603515625 0.19358872031250002 0.65625 0.169263434375 0.66796875 0.1287212953125 0.6533203109375 0.1175722078125 0.5986328125 0.10946377812500001 0.60546875 0.10034179843749999 0.5986328125 0.0831113859375 0.4814453109375 0.077030065625 0.4550781234375 0.0902062625 0.4482421875 0.12162641874999999 0.3759765625 0.152033025 0.3730468765625 0.1631821140625 0.4189453109375 0.227035984375 0.4873046890625 0.2696052328125 0.5302734375 0.2716323390625


================================================
FILE: TumorDetection/train/labels/meningioma_234_jpg.rf.0915b44eb766c431160ad46afd99459b.txt
================================================
2 0.35888671875 0.396484375 0.336819253125 0.4033203140625 0.322881903125 0.4638671875 0.353079490625 0.5048828125 0.4088288828125 0.4716796859375 0.4204433390625 0.4404296859375 0.3914071984375 0.4042968734375 0.35888671875 0.396484375


================================================
FILE: TumorDetection/train/labels/meningioma_242_jpg.rf.bb47982bb3a184a34afaa4af3f515297.txt
================================================
2 0.398185484375 0.5986328125 0.4027103203125 0.5380859359375 0.3744301 0.4921874984375 0.3427562546875 0.490234375 0.2952454875 0.4707031234375 0.2115360375 0.46484375 0.159500434375 0.484375 0.1244329640625 0.5166015625 0.10859604062499999 0.5830078125 0.167418896875 0.6474609359375 0.17420615 0.6787109359375 0.197961534375 0.7011718765625 0.23076658593750002 0.7080078125 0.297507903125 0.693359375 0.339362628125 0.6396484375 0.3721676828125 0.6269531234375 0.398185484375 0.5986328125


================================================
FILE: TumorDetection/train/labels/meningioma_245_jpg.rf.54a22411ef95106132d6b451b999f237.txt
================================================
2 0.5705895875 0.412109375 0.66389354375 0.384765625 0.6937986578124999 0.3564453140625 0.7081531140624999 0.3154296859375 0.7081531140624999 0.2333984375 0.6519314984375 0.185546875 0.604083315625 0.1796875 0.5729819968750001 0.154296875 0.47250080937499994 0.150390625 0.41867160625000005 0.2099609375 0.39713992343750004 0.3623046859375 0.4114943765625 0.3974609375 0.4437919 0.423828125 0.5705895875 0.412109375


================================================
FILE: TumorDetection/train/labels/meningioma_25_jpg.rf.3859c503774a53bb0c6b5b9349562c6a.txt
================================================
2 0.533203125 0.5693359375 0.5498046875 0.583984375 0.5732421875 0.54296875 0.6044921875 0.556640625 0.6123046875 0.5703125 0.6552734375 0.572265625 0.65625 0.5478515625 0.6689453125 0.544921875 0.6962890625 0.51953125 0.7451171875 0.498046875 0.76171875 0.4814453125 0.765625 0.4580078125 0.7421875 0.4267578125 0.73828125 0.4072265625 0.6982421875 0.37890625 0.6826171875 0.384765625 0.66015625 0.4130859375 0.6474609375 0.41796875 0.6162109375 0.412109375 0.5849609375 0.423828125 0.537109375 0.4677734375 0.537109375 0.4912109375 0.5234375 0.5166015625 0.5390625 0.5283203125 0.533203125 0.5693359375


================================================
FILE: TumorDetection/train/labels/meningioma_263_jpg.rf.70ea095866132fa8c4c048652b93160d.txt
================================================
2 0.6376953109375 0.4609375015625 0.7070312484375 0.3818359375 0.7226562484375 0.3330078140625 0.720703125 0.2763671859375 0.705078125 0.2294921859375 0.6552734359375 0.1875 0.5830078140625 0.154296875 0.5146484359375 0.1445312484375 0.4990234359375 0.1796875015625 0.4228515640625 0.1777343734375 0.3847656265625 0.2021484359375 0.341796875 0.2978515640625 0.361328125 0.4423828140625 0.3935546890625 0.4921875015625 0.4208984359375 0.5039062484375 0.4648437515625 0.5048828140625 0.5068359375 0.4941406265625 0.5556640625 0.5 0.5986328140625 0.486328125 0.6376953109375 0.4609375015625


================================================
FILE: TumorDetection/train/labels/meningioma_269_jpg.rf.f4edec265e7781710c266b9f18689699.txt
================================================
2 0.4187681671875 0.3701171875 0.366687865625 0.34375 0.33055050625 0.3046875 0.2731558890625 0.2832031234375 0.263590115625 0.2861328125 0.2699672984375 0.2919921875 0.2625272515625 0.296875 0.24977289374999997 0.296875 0.2465843015625 0.2919921875 0.25508720937499996 0.2880859359375 0.24977289374999997 0.28125 0.24020712031250002 0.2939453125 0.24445857656249997 0.3037109359375 0.22532703281250002 0.3935546875 0.17856104687500002 0.4482421875 0.17218386562500002 0.4873046875 0.23701852968749998 0.546875 0.319921875 0.5722656234375 0.35712209375000004 0.5712890640625 0.3911337203125 0.5458984359375 0.43789971093749996 0.4658203125 0.44427688906249996 0.4111328125 0.4187681671875 0.3701171875


================================================
FILE: TumorDetection/train/labels/meningioma_273_jpg.rf.4f3ebc5e793c8df72fa7e3f2a6ab4300.txt
================================================
2 0.8792348125 0.5244140640625 0.88909170625 0.5263671859375 0.873320678125 0.4873046859375 0.8792348125 0.4814453140625 0.8713492984375 0.4697265640625 0.8772634343749999 0.4658203140625 0.8703636078124999 0.458984375 0.8644494765624999 0.46484375 0.8545925796875 0.4453125 0.8486784484375001 0.44921875 0.8319217296874999 0.4423828140625 0.8319217296874999 0.4326171859375 0.8161507015625 0.4111328140625 0.8181220796875 0.4033203140625 0.8082651875 0.3955078140625 0.8043224296875 0.3544921859375 0.796436915625 0.3466796859375 0.8023510515625001 0.3408203140625 0.7915084671875 0.337890625 0.777708821875 0.31640625 0.7323671125 0.326171875 0.697867990625 0.3525390640625 0.6761828265625 0.3935546859375 0.6702686921875001 0.4482421859375 0.6742114484374999 0.4970703140625 0.7185674671875 0.53515625 0.8072794953125 0.576171875 0.82600759375 0.5751953140625 0.8398072437499999 0.5712890640625 0.840792934375 0.548828125 0.872334990625 0.546875 0.8792348125 0.5244140640625


================================================
FILE: TumorDetection/train/labels/meningioma_280_jpg.rf.d16ce0cd86f7e265c2855bcec0ba02dc.txt
================================================
2 0.5634765625 0.6484375 0.595703125 0.6044921875 0.599609375 0.5361328125 0.5498046875 0.4921875 0.5126953125 0.501953125 0.4765625 0.5556640625 0.478515625 0.6240234375 0.5087890625 0.6484375 0.5341796875 0.65625 0.5634765625 0.6484375


================================================
FILE: TumorDetection/train/labels/meningioma_285_jpg.rf.7ca8f304a112af21e3be1a85ff7013c2.txt
================================================
2 0.5361328125 0.685546875 0.556640625 0.6787109375 0.59765625 0.6318359375 0.6083984375 0.607421875 0.6259765625 0.609375 0.63671875 0.5966796875 0.6337890625 0.5703125 0.6064453125 0.56640625 0.5673828125 0.53515625 0.4990234375 0.537109375 0.4765625 0.5537109375 0.466796875 0.5693359375 0.466796875 0.6103515625 0.5 0.6591796875 0.5361328125 0.685546875


================================================
FILE: TumorDetection/train/labels/meningioma_288_jpg.rf.b74605bf97d174a49dd10f4b8eac6f61.txt
================================================
2 0.6162109375 0.580078125 0.6572265625 0.611328125 0.6796875 0.6083984375 0.642578125 0.5673828125 0.65625 0.5322265625 0.6123046875 0.4453125 0.5791015625 0.4453125 0.5625 0.4619140625 0.5703125 0.4912109375 0.5478515625 0.494140625 0.537109375 0.5126953125 0.5390625 0.5712890625 0.5498046875 0.583984375 0.5869140625 0.587890625 0.6162109375 0.580078125


================================================
FILE: TumorDetection/train/labels/meningioma_294_jpg.rf.00a8bb2e998110e2e598b32410f39c36.txt
================================================
2 0.47433035781250005 0.322265625 0.49665178593750003 0.326171875 0.5189732140625 0.3203125 0.5580357140625 0.2939453125 0.5736607140625 0.2685546875 0.5736607140625 0.2412109375 0.5669642859375 0.2236328125 0.5546875 0.2109375 0.5212053578125 0.1953125 0.45870535781250005 0.201171875 0.444196428125 0.2392578125 0.44196428593750003 0.2822265625 0.446428571875 0.2978515625 0.453125 0.3076171875 0.47433035781250005 0.322265625


================================================
FILE: TumorDetection/train/labels/meningioma_295_jpg.rf.ecfaa49157a1d1af65a80a1e4f3de438.txt
================================================
2 0.431640625 0.3330078125 0.3974609375 0.30859375 0.3701171875 0.302734375 0.3173828125 0.330078125 0.302734375 0.3505859375 0.30859375 0.3896484375 0.34765625 0.4267578125 0.353515625 0.4521484375 0.3681640625 0.46484375 0.384765625 0.4638671875 0.435546875 0.4384765625 0.447265625 0.4091796875 0.447265625 0.3798828125 0.431640625 0.3330078125


================================================
FILE: TumorDetection/train/labels/meningioma_308_jpg.rf.69600117582e60b8e7cf3652c3f8723c.txt
================================================
2 0.62890625 0.5693359375 0.634765625 0.5322265625 0.6044921875 0.474609375 0.5693359375 0.466796875 0.53515625 0.4833984375 0.5390625 0.5361328125 0.5693359375 0.580078125 0.6123046875 0.580078125 0.62890625 0.5693359375


================================================
FILE: TumorDetection/train/labels/meningioma_310_jpg.rf.324dc73989291cfcacf2f7ff6f1f9b71.txt
================================================
2 0.412109375 0.3212890625 0.3955078125 0.3046875 0.3505859375 0.287109375 0.3095703125 0.296875 0.294921875 0.3095703125 0.263671875 0.4208984375 0.2978515625 0.486328125 0.337890625 0.4912109375 0.3564453125 0.486328125 0.3984375 0.4482421875 0.421875 0.3701171875 0.412109375 0.3212890625


================================================
FILE: TumorDetection/train/labels/meningioma_312_jpg.rf.556f795ceb5e9714556ad461c6ee13cc.txt
================================================
2 0.3740234375 0.298828125 0.3544921875 0.302734375 0.3203125 0.3349609375 0.298828125 0.4072265625 0.302734375 0.4306640625 0.296875 0.4404296875 0.3251953125 0.46875 0.35546875 0.4775390625 0.38671875 0.4619140625 0.41796875 0.3935546875 0.41796875 0.3251953125 0.3740234375 0.298828125


================================================
FILE: TumorDetection/train/labels/meningioma_316_jpg.rf.0c8dc01bd2d691310062457ae79ff838.txt
================================================
2 0.6826171875 0.7109375 0.697265625 0.7021484375 0.748046875 0.6181640625 0.76171875 0.5576171875 0.7333984375 0.53515625 0.6728515625 0.52734375 0.65234375 0.5380859375 0.6015625 0.6005859375 0.595703125 0.6396484375 0.609375 0.6904296875 0.6435546875 0.7109375 0.6826171875 0.7109375


================================================
FILE: TumorDetection/train/labels/meningioma_318_jpg.rf.497f2dcaefcfdfe088110cffe3a48de1.txt
================================================
2 0.6533203125 0.5546875 0.6240234375 0.55859375 0.6015625 0.5791015625 0.6015625 0.6123046875 0.6484375 0.6728515625 0.669921875 0.6806640625 0.703125 0.6474609375 0.7109375 0.6064453125 0.6865234375 0.568359375 0.6533203125 0.5546875


================================================
FILE: TumorDetection/train/labels/meningioma_319_jpg.rf.9cc88ece353c5431b9f49d7ed6c83ad1.txt
================================================
2 0.6162109375 0.57421875 0.60546875 0.6025390625 0.6171875 0.6240234375 0.6416015625 0.642578125 0.66796875 0.6416015625 0.697265625 0.6103515625 0.68359375 0.5751953125 0.6513671875 0.564453125 0.6162109375 0.57421875


================================================
FILE: TumorDetection/train/labels/meningioma_325_jpg.rf.cecf06691481e910dde208a823f0d390.txt
================================================
2 0.5302734375 0.49609375 0.5693359375 0.48828125 0.5986328125 0.47265625 0.6328125 0.4287109375 0.62890625 0.3857421875 0.5947265625 0.345703125 0.5751953125 0.3359375 0.5478515625 0.337890625 0.4990234375 0.359375 0.484375 0.3935546875 0.482421875 0.4130859375 0.51171875 0.4677734375 0.51171875 0.4814453125 0.5302734375 0.49609375


================================================
FILE: TumorDetection/train/labels/meningioma_331_jpg.rf.c123e69b071dd856fd282204f9ec4e36.txt
================================================
2 0.46484375 0.7841796875 0.46875 0.7470703125 0.4423828125 0.70703125 0.4033203125 0.697265625 0.3740234375 0.7109375 0.365234375 0.7275390625 0.365234375 0.7744140625 0.3759765625 0.79296875 0.4111328125 0.80078125 0.46484375 0.7841796875


================================================
FILE: TumorDetection/train/labels/meningioma_342_jpg.rf.30a0372c2bafc1ec4a44fb86c84c1595.txt
================================================
2 0.4521484375 0.546875 0.4833984375 0.541015625 0.5078125 0.5224609375 0.525390625 0.4892578125 0.53125 0.3916015625 0.5234375 0.3115234375 0.3955078125 0.345703125 0.357421875 0.3779296875 0.333984375 0.4169921875 0.337890625 0.4619140625 0.3740234375 0.484375 0.4033203125 0.525390625 0.4521484375 0.546875


================================================
FILE: TumorDetection/train/labels/meningioma_343_jpg.rf.8088aa8dac38e2d92a21c145367ed44f.txt
================================================
2 0.4267578125 0.52734375 0.4697265625 0.529296875 0.509765625 0.5166015625 0.513671875 0.4306640625 0.5234375 0.4033203125 0.5107421875 0.314453125 0.4638671875 0.333984375 0.4365234375 0.33203125 0.3779296875 0.353515625 0.361328125 0.3720703125 0.3515625 0.3994140625 0.34765625 0.4404296875 0.357421875 0.4619140625 0.4072265625 0.49609375 0.4267578125 0.52734375


================================================
FILE: TumorDetection/train/labels/meningioma_346_jpg.rf.af45eb1c56f185c1b4fce658d886cb91.txt
================================================
2 0.685546875 0.6279296875 0.7294921875 0.6640625 0.7734375 0.6005859375 0.78125 0.5986328125 0.779296875 0.6123046875 0.7861328125 0.6171875 0.8515625 0.5361328125 0.8671875 0.4853515625 0.853515625 0.3916015625 0.8173828125 0.3359375 0.7978515625 0.3203125 0.7744140625 0.31640625 0.76171875 0.3232421875 0.751953125 0.3447265625 0.75 0.3642578125 0.759765625 0.3916015625 0.7548828125 0.396484375 0.7353515625 0.384765625 0.7021484375 0.3984375 0.66015625 0.4482421875 0.65625 0.4794921875 0.666015625 0.5107421875 0.669921875 0.5947265625 0.685546875 0.6279296875


================================================
FILE: TumorDetection/train/labels/meningioma_347_jpg.rf.86020889954b4fd005848f8754283ac6.txt
================================================
2 0.8310546875 0.5625 0.84765625 0.5498046875 0.873046875 0.4990234375 0.859375 0.4013671875 0.83984375 0.3525390625 0.8193359375 0.33203125 0.7783203125 0.314453125 0.7509765625 0.31640625 0.732421875 0.3408203125 0.751953125 0.3935546875 0.7197265625 0.369140625 0.7021484375 0.373046875 0.67578125 0.3994140625 0.646484375 0.4716796875 0.646484375 0.4931640625 0.66015625 0.5185546875 0.669921875 0.6083984375 0.6982421875 0.6484375 0.7265625 0.6630859375 0.7783203125 0.599609375 0.80859375 0.5849609375 0.8154296875 0.560546875 0.8310546875 0.5625


================================================
FILE: TumorDetection/train/labels/meningioma_355_jpg.rf.239a9f8e75698f9fbc93ab2fc445a9e0.txt
================================================
2 0.8017578125 0.5825213875 0.8408203125 0.5825213875 0.8515625 0.575633465625 0.84375 0.5677615546875 0.853515625 0.565793578125 0.849609375 0.54020986875 0.8554687515625 0.520530090625 0.8632812484375 0.5185621140625 0.8554687515625 0.5047862703125 0.865234375 0.5008503156249999 0.8554687515625 0.4988823359375 0.849609375 0.47526660468750004 0.8554687515625 0.4693626734375 0.8447265625 0.460506771875 0.8515625 0.4536188515625 0.8310546875 0.44673092968750006 0.8398437515625 0.4398430078125 0.8310546875 0.4290191296875 0.8398437515625 0.42409918593749996 0.833984375 0.41229131875 0.8398437515625 0.404419409375 0.8291015625 0.395563509375 0.8398437515625 0.39064356718750004 0.8232421875 0.381787665625 0.8291015625 0.37391575625 0.7958984375 0.36604384375 0.7851562484375 0.34931603437500003 0.7822265625 0.3247163140625 0.7451171875 0.3266842921875 0.7158203125 0.340460134375 0.693359375 0.363091878125 0.671875 0.408355365625 0.671875 0.487074471875 0.6943359375 0.5215140796875 0.7275390625 0.547097790625 0.8017578125 0.5825213875


================================================
FILE: TumorDetection/train/labels/meningioma_35_jpg.rf.7c99aa29547ebcb3a8cb8a78c95ad76a.txt
================================================
2 0.6650390625 0.287109375 0.6416015625 0.27734375 0.5693359375 0.33984375 0.54296875 0.3759765625 0.5234375 0.4150390625 0.515625 0.4677734375 0.521484375 0.4951171875 0.5654296875 0.546875 0.6181640625 0.576171875 0.658203125 0.5791015625 0.7685546875 0.50390625 0.7734375 0.4892578125 0.720703125 0.3408203125 0.6650390625 0.287109375


================================================
FILE: TumorDetection/train/labels/meningioma_368_jpg.rf.9d74c6881f81c543b1c9a5cef74a6537.txt
================================================
2 0.50390625 0.5166015625 0.50390625 0.4990234375 0.4619140625 0.45703125 0.4228515625 0.455078125 0.3583984375 0.474609375 0.349609375 0.4873046875 0.341796875 0.5205078125 0.37109375 0.5673828125 0.3671875 0.5947265625 0.3759765625 0.60546875 0.392578125 0.6083984375 0.412109375 0.5869140625 0.41015625 0.5439453125 0.4267578125 0.517578125 0.50390625 0.5166015625


================================================
FILE: TumorDetection/train/labels/meningioma_371_jpg.rf.6ed100fc09c2a6066a7a698a7a861797.txt
================================================
2 0.748046875 0.3369140625 0.6669921875 0.279296875 0.6240234375 0.28515625 0.5849609375 0.27734375 0.5712890625 0.283203125 0.53515625 0.3505859375 0.541015625 0.3759765625 0.57421875 0.3974609375 0.5927734375 0.43359375 0.6123046875 0.439453125 0.6318359375 0.45703125 0.703125 0.4658203125 0.73046875 0.4541015625 0.76171875 0.4052734375 0.765625 0.3720703125 0.748046875 0.3369140625


================================================
FILE: TumorDetection/train/labels/meningioma_375_jpg.rf.380d15275cfa0e4d376525ae9c86a507.txt
================================================
2 0.2919921875 0.427734375 0.3466796875 0.431640625 0.376953125 0.4111328125 0.384765625 0.3662109375 0.40234375 0.3564453125 0.412109375 0.3212890625 0.3984375 0.2998046875 0.404296875 0.2861328125 0.400390625 0.2685546875 0.3720703125 0.2421875 0.3017578125 0.2265625 0.263671875 0.2607421875 0.23046875 0.3291015625 0.228515625 0.3681640625 0.2412109375 0.41796875 0.2744140625 0.4140625 0.2919921875 0.427734375


================================================
FILE: TumorDetection/train/labels/meningioma_384_jpg.rf.1156717de547acc9888b45144028e2f3.txt
================================================
2 0.5146484375 0.3359375 0.525390625 0.3193359375 0.50390625 0.2998046875 0.5 0.2763671875 0.4697265625 0.24609375 0.4384765625 0.2421875 0.4033203125 0.255859375 0.390625 0.2685546875 0.380859375 0.2822265625 0.375 0.3154296875 0.38671875 0.3544921875 0.4072265625 0.37890625 0.4345703125 0.390625 0.455078125 0.3896484375 0.4990234375 0.37890625 0.509765625 0.3642578125 0.501953125 0.3388671875 0.5146484375 0.3359375


================================================
FILE: TumorDetection/train/labels/meningioma_386_jpg.rf.837a2af568f580f346a797f1bd5015cc.txt
================================================
2 0.4794921875 0.3828125 0.509765625 0.3623046875 0.505859375 0.3291015625 0.515625 0.3154296875 0.513671875 0.2958984375 0.501953125 0.2626953125 0.4794921875 0.2421875 0.4521484375 0.240234375 0.4052734375 0.25390625 0.3828125 0.2822265625 0.3828125 0.3212890625 0.4208984375 0.369140625 0.4384765625 0.37890625 0.4794921875 0.3828125


================================================
FILE: TumorDetection/train/labels/meningioma_391_jpg.rf.c05273c79dfe211092dbc147bc57f96c.txt
================================================
2 0.48828125 0.5810546875 0.4697265625 0.564453125 0.4423828125 0.556640625 0.3720703125 0.578125 0.349609375 0.5966796875 0.337890625 0.6220703125 0.3359375 0.6591796875 0.34765625 0.6943359375 0.3720703125 0.716796875 0.45703125 0.7216796875 0.4775390625 0.7109375 0.4921875 0.6865234375 0.498046875 0.6103515625 0.48828125 0.5810546875


================================================
FILE: TumorDetection/train/labels/meningioma_392_jpg.rf.a7e384c4a7b1146a935811a2c04e47b4.txt
================================================
2 0.3984375 0.7041015625 0.4033203125 0.7109375 0.4404296875 0.72265625 0.484375 0.6962890625 0.484375 0.6357421875 0.4921875 0.6083984375 0.486328125 0.5888671875 0.4697265625 0.572265625 0.4208984375 0.56640625 0.3837890625 0.58203125 0.36328125 0.5986328125 0.3623046875 0.609375 0.3564453125 0.60546875 0.3515625 0.6123046875 0.353515625 0.6435546875 0.3994140625 0.6953125 0.41015625 0.6982421875 0.3984375 0.7041015625


================================================
FILE: TumorDetection/train/labels/meningioma_398_jpg.rf.f303a581bba88e387c4a1ab8dbbdd418.txt
================================================
2 0.724609375 0.3505859375 0.6943359375 0.33984375 0.6611328125 0.34375 0.640625 0.3642578125 0.64453125 0.3837890625 0.6630859375 0.404296875 0.693359375 0.4052734375 0.7119140625 0.404296875 0.728515625 0.3935546875 0.734375 0.3701171875 0.724609375 0.3505859375


================================================
FILE: TumorDetection/train/labels/meningioma_3_jpg.rf.3d6ab426da351985af9403198a44e6f0.txt
================================================
2 0.37109375 0.5771484375 0.3701171875 0.54296875 0.3447265625 0.533203125 0.3271484375 0.513671875 0.2978515625 0.51171875 0.275390625 0.5302734375 0.2734375 0.5810546875 0.3046875 0.6748046875 0.328125 0.6865234375 0.34375 0.6337890625 0.3759765625 0.61328125 0.38671875 0.5947265625 0.37109375 0.5771484375


================================================
FILE: TumorDetection/train/labels/meningioma_401_jpg.rf.e82a83f493a3fdbb2c7f4ec1fb0c9397.txt
================================================
2 0.7255859375 0.322265625 0.7060546875 0.322265625 0.6728515625 0.333984375 0.66015625 0.3486328125 0.658203125 0.3681640625 0.669921875 0.4150390625 0.6923828125 0.431640625 0.720703125 0.4306640625 0.7421875 0.4130859375 0.73046875 0.3603515625 0.744140625 0.3330078125 0.7255859375 0.322265625


================================================
FILE: TumorDetection/train/labels/meningioma_410_jpg.rf.ff44de7d5514fa24f683f3fbdc2ddea2.txt
================================================
2 0.541015625 0.1962890625 0.5341796875 0.185546875 0.4658203125 0.1796875 0.4033203125 0.203125 0.369140625 0.2255859375 0.373046875 0.2490234375 0.390625 0.2607421875 0.3974609375 0.287109375 0.4521484375 0.30078125 0.474609375 0.3154296875 0.5009765625 0.310546875 0.525390625 0.2919921875 0.529296875 0.2177734375 0.541015625 0.1962890625


================================================
FILE: TumorDetection/train/labels/meningioma_412_jpg.rf.e9cb10656771b59ce7c885fd2bfb6665.txt
================================================
2 0.4150390625 0.1875 0.37109375 0.2060546875 0.37109375 0.2587890625 0.3916015625 0.294921875 0.46875 0.2939453125 0.5078125 0.2607421875 0.5078125 0.2333984375 0.4912109375 0.19140625 0.4150390625 0.1875


================================================
FILE: TumorDetection/train/labels/meningioma_415_jpg.rf.8960d1e0deb57ffc78750398cb7410f5.txt
================================================
2 0.45 0.30390625 0.14375 0.1421875
2 0.5078125 0.2568359375 0.4990234375 0.248046875 0.4560546875 0.25 0.4326171875 0.2421875 0.392578125 0.2705078125 0.38671875 0.3017578125 0.400390625 0.3427734375 0.4267578125 0.365234375 0.455078125 0.3701171875 0.4814453125 0.36328125 0.501953125 0.3310546875 0.51171875 0.3095703125 0.5078125 0.2568359375


================================================
FILE: TumorDetection/train/labels/meningioma_424_jpg.rf.6e971bda062dbe09e62cc3939de38b3f.txt
================================================
2 0.5569196421875 0.3489118296875 0.5862165171875 0.3743024546875 0.5901227671875 0.3977399546875 0.6457868296875 0.44754464218749995 0.6926618296875 0.46707589218749995 0.7160993296875 0.46316964218749995 0.7268415171875 0.44070870468749995 0.7170758921875 0.42117745468749995 0.7659040171875 0.40457589218749995 0.7589285718750001 0.37946428593750003 0.7561383921875 0.35435267812500004 0.7346540171875 0.3391462046875 0.7190290171875 0.3332868296875 0.7170758921875 0.3078962046875 0.6731305796875 0.2815290171875 0.6487165171875 0.2453962046875 0.6184430796875 0.2268415171875 0.5813337046875 0.2287946421875 0.5647321421875 0.2453962046875 0.5666852671875 0.2629743296875 0.5549665171875 0.3020368296875 0.5608258921875 0.3137555796875 0.5569196421875 0.3489118296875


================================================
FILE: TumorDetection/train/labels/meningioma_427_jpg.rf.912d9eb73f6ae6d1019c74b3b8354217.txt
================================================
2 0.6142578125 0.412109375 0.6650390625 0.431640625 0.7060546875 0.431640625 0.7470703125 0.41015625 0.771484375 0.3818359375 0.75390625 0.3291015625 0.73828125 0.3056640625 0.6455078125 0.216796875 0.6279296875 0.216796875 0.5849609375 0.19921875 0.5673828125 0.201171875 0.533203125 0.2451171875 0.51171875 0.2998046875 0.51953125 0.3720703125 0.529296875 0.3896484375 0.5556640625 0.41015625 0.6142578125 0.412109375


================================================
FILE: TumorDetection/train/labels/meningioma_428_jpg.rf.161b483f1c43282a19796f3db11ba607.txt
================================================
2 0.7578125 0.3798828125 0.767578125 0.3466796875 0.75 0.3076171875 0.6884765625 0.2421875 0.6708984375 0.2265625 0.5966796875 0.19140625 0.5693359375 0.19921875 0.521484375 0.2587890625 0.505859375 0.3037109375 0.509765625 0.3408203125 0.5 0.3564453125 0.5 0.3759765625 0.5205078125 0.40234375 0.5673828125 0.427734375 0.587890625 0.4267578125 0.6298828125 0.4140625 0.6826171875 0.42578125 0.7119140625 0.423828125 0.7578125 0.3798828125


================================================
FILE: TumorDetection/train/labels/meningioma_446_jpg.rf.586ea523102745fbd6d7d19556a4f4a9.txt
================================================
2 0.4560546875 0.14453125 0.4287109375 0.14453125 0.388671875 0.1669921875 0.37109375 0.2333984375 0.384765625 0.2724609375 0.4013671875 0.2890625 0.4287109375 0.30078125 0.4609375 0.3017578125 0.515625 0.2685546875 0.5234375 0.2236328125 0.4951171875 0.16796875 0.4560546875 0.14453125


================================================
FILE: TumorDetection/train/labels/meningioma_449_jpg.rf.8add6f9e9b0a0427a42f614fa15f4873.txt
================================================
2 0.4541015625 0.185546875 0.4248046875 0.177734375 0.390625 0.2041015625 0.38671875 0.2626953125 0.4267578125 0.296875 0.46484375 0.2978515625 0.50390625 0.2666015625 0.50390625 0.1884765625 0.4931640625 0.181640625 0.4541015625 0.185546875


================================================
FILE: TumorDetection/train/labels/meningioma_460_jpg.rf.0faac894ebf5dec9e5112ced6dc5bbb2.txt
================================================
2 0.470703125 0.1650390625 0.4462890625 0.1328125 0.4287109375 0.126953125 0.3271484375 0.166015625 0.291015625 0.2060546875 0.2890625 0.2490234375 0.3134765625 0.283203125 0.3720703125 0.3046875 0.400390625 0.3056640625 0.4580078125 0.2734375 0.47265625 0.2412109375 0.4765625 0.2001953125 0.470703125 0.1650390625


================================================
FILE: TumorDetection/train/labels/meningioma_466_jpg.rf.59370219c28541ae27c863a2a39e37d3.txt
================================================
2 0.494140625 0.2041015625 0.4912109375 0.1875 0.4814453125 0.193359375 0.4736328125 0.1796875 0.4404296875 0.1640625 0.3837890625 0.1640625 0.3291015625 0.193359375 0.30078125 0.2197265625 0.3046875 0.2412109375 0.3232421875 0.267578125 0.3642578125 0.28125 0.400390625 0.2802734375 0.4345703125 0.279296875 0.45703125 0.2646484375 0.47265625 0.2158203125 0.494140625 0.2041015625


================================================
FILE: TumorDetection/train/labels/meningioma_46_jpg.rf.259eebd958cd27bed31605e0c64f2f20.txt
================================================
2 0.59375 0.5849609375 0.60546875 0.5419921875 0.5830078125 0.50390625 0.5673828125 0.49609375 0.5205078125 0.50390625 0.478515625 0.5341796875 0.474609375 0.5556640625 0.5 0.5947265625 0.5078125 0.6298828125 0.5185546875 0.615234375 0.5419921875 0.625 0.5654296875 0.62109375 0.59375 0.5849609375


================================================
FILE: TumorDetection/train/labels/meningioma_473_jpg.rf.064ae57ceafe17ae32c5e4b6792bac84.txt
================================================
2 0.5595703125 0.115234375 0.5029296875 0.119140625 0.49609375 0.1298828125 0.494140625 0.2060546875 0.5302734375 0.224609375 0.56640625 0.2255859375 0.603515625 0.1904296875 0.609375 0.1728515625 0.603515625 0.1396484375 0.5595703125 0.115234375


================================================
FILE: TumorDetection/train/labels/meningioma_474_jpg.rf.3489d5edbb9f106b58990bec025051c4.txt
================================================
2 0.5673828125 0.126953125 0.5009765625 0.134765625 0.49609375 0.1884765625 0.5185546875 0.20703125 0.55078125 0.2119140625 0.5859375 0.1884765625 0.59375 0.1650390625 0.58984375 0.1396484375 0.5673828125 0.126953125


================================================
FILE: TumorDetection/train/labels/meningioma_477_jpg.rf.44cfb7a6c54822440200eded86616949.txt
================================================
2 0.5830078125 0.31640625 0.6767578125 0.275390625 0.697265625 0.2568359375 0.6953125 0.1845703125 0.6123046875 0.123046875 0.5615234375 0.111328125 0.5224609375 0.115234375 0.5078125 0.1533203125 0.49609375 0.2548828125 0.513671875 0.2841796875 0.5478515625 0.3125 0.5830078125 0.31640625


================================================
FILE: TumorDetection/train/labels/meningioma_482_jpg.rf.04fa49e7085ab5dde5a0a4a1bc568640.txt
================================================
2 0.5009765625 0.509765625 0.546875 0.4833984375 0.5546875 0.4580078125 0.55078125 0.4287109375 0.5322265625 0.40625 0.5048828125 0.390625 0.4765625 0.3935546875 0.486328125 0.4052734375 0.4765625 0.4072265625 0.48046875 0.4208984375 0.46875 0.4248046875 0.4677734375 0.4453125 0.46484375 0.4150390625 0.474609375 0.3955078125 0.4677734375 0.388671875 0.4384765625 0.39453125 0.4365234375 0.41796875 0.4248046875 0.412109375 0.3916015625 0.423828125 0.3525390625 0.42578125 0.306640625 0.5048828125 0.33984375 0.5146484375 0.34765625 0.5634765625 0.3623046875 0.58203125 0.443359375 0.6025390625 0.462890625 0.5966796875 0.462890625 0.5556640625 0.470703125 0.5439453125 0.5009765625 0.509765625


================================================
FILE: TumorDetection/train/labels/meningioma_484_jpg.rf.b0dacc7c4c7d73ebaf772cea4de6c25f.txt
================================================
2 0.494140625 0.5107421875 0.5361328125 0.5 0.5625 0.4638671875 0.5380859375 0.43359375 0.5126953125 0.419921875 0.4560546875 0.41015625 0.4306640625 0.396484375 0.3896484375 0.40234375 0.353515625 0.4228515625 0.34765625 0.4462890625 0.318359375 0.4736328125 0.3046875 0.5146484375 0.3330078125 0.52734375 0.3583984375 0.52734375 0.388671875 0.5537109375 0.3916015625 0.564453125 0.4013671875 0.5546875 0.4130859375 0.5625 0.41015625 0.5556640625 0.4208984375 0.552734375 0.41796875 0.5595703125 0.4296875 0.5654296875 0.419921875 0.5751953125 0.4365234375 0.59375 0.4453125 0.5927734375 0.4951171875 0.55078125 0.525390625 0.5380859375 0.5029296875 0.5390625 0.494140625 0.5107421875


================================================
FILE: TumorDetection/train/labels/meningioma_489_jpg.rf.37968ddddd2f75afc081ab00dde4b5f7.txt
================================================
2 0.6708984375 0.1484375 0.6083984375 0.140625 0.591796875 0.1708984375 0.591796875 0.1904296875 0.6142578125 0.21875 0.65234375 0.2314453125 0.6689453125 0.228515625 0.6875 0.2080078125 0.689453125 0.1728515625 0.6708984375 0.1484375


================================================
FILE: TumorDetection/train/labels/meningioma_502_jpg.rf.c6df0ca015dbbe0d5e7633f9c0d182ee.txt
================================================
2 0.3720703125 0.42578125 0.3544921875 0.435546875 0.333984375 0.4755859375 0.3583984375 0.47265625 0.412109375 0.5029296875 0.443359375 0.4853515625 0.439453125 0.4638671875 0.3994140625 0.427734375 0.3720703125 0.42578125


================================================
FILE: TumorDetection/train/labels/meningioma_513_jpg.rf.9ccd3e3625158e49d46d039ad5808006.txt
================================================
2 0.4150390625 0.32421875 0.3759765625 0.341796875 0.3515625 0.3681640625 0.349609375 0.4013671875 0.3798828125 0.4375 0.42578125 0.4462890625 0.451171875 0.4208984375 0.4609375 0.3525390625 0.4365234375 0.326171875 0.4150390625 0.32421875


================================================
FILE: TumorDetection/train/labels/meningioma_517_jpg.rf.10c6b3b1eaec17a29c6747d5d7ff1d64.txt
================================================
2 0.4990234375 0.2109375 0.4765625 0.2470703125 0.48828125 0.2861328125 0.4951171875 0.29296875 0.513671875 0.2919921875 0.5390625 0.2822265625 0.546875 0.2529296875 0.5263671875 0.220703125 0.4990234375 0.2109375


================================================
FILE: TumorDetection/train/labels/meningioma_518_jpg.rf.299c80c7b531ee4bc9516f9d56b9d8a6.txt
================================================
2 0.4814453125 0.240234375 0.474609375 0.2548828125 0.48046875 0.2939453125 0.4951171875 0.302734375 0.517578125 0.3017578125 0.54296875 0.2861328125 0.552734375 0.2509765625 0.5361328125 0.2421875 0.4814453125 0.240234375


================================================
FILE: TumorDetection/train/labels/meningioma_520_jpg.rf.7fbfefe2feacebd4d94756793fc247c8.txt
================================================
2 0.5703125 0.3271484375 0.529296875 0.2685546875 0.509765625 0.2158203125 0.51171875 0.2021484375 0.4892578125 0.189453125 0.4677734375 0.189453125 0.4130859375 0.203125 0.3603515625 0.228515625 0.3203125 0.2822265625 0.326171875 0.3427734375 0.3720703125 0.373046875 0.4033203125 0.376953125 0.4365234375 0.41796875 0.515625 0.4365234375 0.5478515625 0.42578125 0.578125 0.3974609375 0.583984375 0.3662109375 0.5703125 0.3271484375


================================================
FILE: TumorDetection/train/labels/meningioma_523_jpg.rf.517e6cbcb701c14604ae34affef7c00c.txt
================================================
2 0.5712890625 0.37890625 0.564453125 0.3798828125 0.568359375 0.3525390625 0.548828125 0.3232421875 0.53125 0.3115234375 0.4921875 0.1669921875 0.4794921875 0.15625 0.4462890625 0.15625 0.4140625 0.1884765625 0.3828125 0.2392578125 0.376953125 0.2900390625 0.38671875 0.3447265625 0.3984375 0.3681640625 0.4296875 0.3916015625 0.4462890625 0.41796875 0.4833984375 0.435546875 0.509765625 0.4345703125 0.56640625 0.4033203125 0.564453125 0.3935546875 0.57421875 0.3876953125 0.5712890625 0.37890625


================================================
FILE: TumorDetection/train/labels/meningioma_526_jpg.rf.98e109530bc9b3ede3d6e6cd4a06050e.txt
================================================
2 0.87109375 0.3994140625 0.8779296875 0.408203125 0.884765625 0.3994140625 0.884765625 0.3349609375 0.8798828125 0.34765625 0.875 0.3349609375 0.8740234375 0.34765625 0.87109375 0.3037109375 0.8134765625 0.259765625 0.7724609375 0.2421875 0.6884765625 0.234375 0.642578125 0.2626953125 0.666015625 0.4169921875 0.7294921875 0.484375 0.7587890625 0.494140625 0.7880859375 0.443359375 0.794921875 0.4541015625 0.791015625 0.5166015625 0.818359375 0.4931640625 0.83984375 0.4501953125 0.86328125 0.4287109375 0.87109375 0.3994140625


================================================
FILE: TumorDetection/train/labels/meningioma_540_jpg.rf.c7dba9ffd13a4576620ef3aeb2b60ea8.txt
================================================
2 0.724609375 0.1611328125 0.6806640625 0.134765625 0.5986328125 0.1171875 0.5537109375 0.115234375 0.4833984375 0.125 0.47265625 0.1416015625 0.466796875 0.1904296875 0.498046875 0.2607421875 0.5302734375 0.287109375 0.5712890625 0.306640625 0.615234375 0.3115234375 0.6474609375 0.302734375 0.708984375 0.2548828125 0.736328125 0.1962890625 0.724609375 0.1611328125


================================================
FILE: TumorDetection/train/labels/meningioma_543_jpg.rf.e11bae3ffd29e3d4d0394811c4dab3d3.txt
================================================
2 0.716796875 0.6181640625 0.6806640625 0.595703125 0.6376953125 0.58984375 0.609375 0.6103515625 0.6015625 0.6572265625 0.6357421875 0.685546875 0.6640625 0.6884765625 0.6884765625 0.6875 0.7109375 0.6767578125 0.720703125 0.6474609375 0.716796875 0.6181640625


================================================
FILE: TumorDetection/train/labels/meningioma_544_jpg.rf.4f027eb4b90a9d8ce0b3d0370b2d0ee8.txt
================================================
2 0.6552734375 0.58984375 0.6298828125 0.59375 0.607421875 0.6103515625 0.603515625 0.6455078125 0.6220703125 0.6796875 0.6640625 0.6826171875 0.7197265625 0.6640625 0.720703125 0.6279296875 0.6845703125 0.599609375 0.6552734375 0.58984375


================================================
FILE: TumorDetection/train/labels/meningioma_547_jpg.rf.ce87498c81d59d6499864d02df593180.txt
================================================
2 0.4150390625 0.302734375 0.4345703125 0.291015625 0.466796875 0.2509765625 0.4609375 0.2080078125 0.4248046875 0.181640625 0.3876953125 0.1875 0.337890625 0.2314453125 0.333984375 0.2470703125 0.34375 0.2822265625 0.3857421875 0.306640625 0.4150390625 0.302734375


================================================
FILE: TumorDetection/train/labels/meningioma_55_jpg.rf.ef13fdd7d8b8b5c722d124c14792f38c.txt
================================================
2 0.744140625 0.4052734375 0.72265625 0.3955078125 0.72265625 0.3603515625 0.701171875 0.3388671875 0.705078125 0.3271484375 0.6611328125 0.294921875 0.6396484375 0.29296875 0.548828125 0.3505859375 0.548828125 0.3740234375 0.53515625 0.3955078125 0.53125 0.4580078125 0.541015625 0.4794921875 0.5615234375 0.49609375 0.654296875 0.5185546875 0.7099609375 0.498046875 0.7587890625 0.46484375 0.76171875 0.4482421875 0.744140625 0.4052734375


================================================
FILE: TumorDetection/train/labels/meningioma_561_jpg.rf.544bf3fac642cba5ff23b9ec6f8fa39d.txt
================================================
2 0.748046875 0.3173828125 0.76171875 0.2939453125 0.7392578125 0.2578125 0.6962890625 0.232421875 0.6650390625 0.2421875 0.65234375 0.2685546875 0.650390625 0.2919921875 0.6787109375 0.330078125 0.7294921875 0.330078125 0.748046875 0.3173828125


================================================
FILE: TumorDetection/train/labels/meningioma_564_jpg.rf.b6b6cf99462359a70869fe1bb7fa9fa6.txt
================================================
2 0.6845703125 0.482421875 0.7138671875 0.482421875 0.7392578125 0.470703125 0.755859375 0.4501953125 0.76171875 0.4091796875 0.775390625 0.3818359375 0.775390625 0.3525390625 0.755859375 0.3134765625 0.7255859375 0.29296875 0.6904296875 0.283203125 0.6455078125 0.2890625 0.603515625 0.3232421875 0.5859375 0.3544921875 0.587890625 0.4111328125 0.6142578125 0.423828125 0.6591796875 0.47265625 0.6845703125 0.482421875


================================================
FILE: TumorDetection/train/labels/meningioma_567_jpg.rf.a077d99c54ffc0d725d9c939a64ce443.txt
================================================
2 0.6943359375 0.458984375 0.7333984375 0.439453125 0.755859375 0.4150390625 0.76171875 0.3388671875 0.7392578125 0.30078125 0.7021484375 0.296875 0.6396484375 0.318359375 0.603515625 0.3642578125 0.6015625 0.4033203125 0.6474609375 0.447265625 0.6748046875 0.4609375 0.6943359375 0.458984375


================================================
FILE: TumorDetection/train/labels/meningioma_56_jpg.rf.809e46b330e6d1e7c74a8b41330f8f6b.txt
================================================
2 0.552734375 0.4521484375 0.55078125 0.4716796875 0.5654296875 0.490234375 0.6689453125 0.501953125 0.744140625 0.4599609375 0.734375 0.4169921875 0.70703125 0.3876953125 0.705078125 0.3603515625 0.6748046875 0.326171875 0.6416015625 0.310546875 0.5634765625 0.333984375 0.55078125 0.3486328125 0.55859375 0.3837890625 0.544921875 0.3994140625 0.54296875 0.4208984375 0.552734375 0.4521484375


================================================
FILE: TumorDetection/train/labels/meningioma_584_jpg.rf.3d0ed1863e116fea7147a0411f1bab22.txt
================================================
2 0.31016989375000004 0.259765625 0.2673878375 0.23828125 0.2032147578125 0.2265625015625 0.17588177968750002 0.2958984375 0.196084415625 0.328125 0.2210406140625 0.3310546890625 0.18776568437500002 0.3505859359375 0.17588177968750002 0.3916015625 0.1283461625 0.4365234375 0.1117086984375 0.5244140640625 0.136664896875 0.546875 0.2257941765625 0.5478515625 0.3030395515625 0.5175781234375 0.39454561093749996 0.4482421875 0.4159366390625 0.3583984375 0.31016989375000004 0.259765625


================================================
FILE: TumorDetection/train/labels/meningioma_585_jpg.rf.5df72b5d661e615e520bf19ed05752f1.txt
================================================
2 0.458414396875 0.72265625 0.42071984375 0.7529296875 0.4182879375 0.8115234375 0.45111867656250004 0.845703125 0.49610895 0.8466796875 0.5204280156250001 0.8291015625 0.5277237359375 0.7861328125 0.50340466875 0.7470703125 0.458414396875 0.72265625


================================================
FILE: TumorDetection/train/labels/meningioma_588_jpg.rf.2b39f2bd67146ad3088d621a41d0a05e.txt
================================================
2 0.630859375 0.5185546875 0.6328125 0.5654296875 0.6513671875 0.576171875 0.6962890625 0.576171875 0.71484375 0.5927734375 0.712890625 0.5283203125 0.6787109375 0.4921875 0.6552734375 0.4921875 0.630859375 0.5185546875


================================================
FILE: TumorDetection/train/labels/meningioma_602_jpg.rf.b3e6e4fd5cead908876400224522aa10.txt
================================================
2 0.5068359375 0.306640625 0.5517578125 0.279296875 0.576171875 0.2412109375 0.56640625 0.1962890625 0.5537109375 0.181640625 0.5380859375 0.177734375 0.4619140625 0.1796875 0.3994140625 0.201171875 0.38671875 0.2177734375 0.388671875 0.2705078125 0.4208984375 0.30078125 0.4794921875 0.29296875 0.5068359375 0.306640625


================================================
FILE: TumorDetection/train/labels/meningioma_61_jpg.rf.2ccd18e873063f2ed8d54852e9f66835.txt
================================================
2 0.4736328125 0.70703125 0.5234375 0.6826171875 0.537109375 0.6416015625 0.53515625 0.6083984375 0.5244140625 0.595703125 0.4755859375 0.58203125 0.4521484375 0.583984375 0.4208984375 0.591796875 0.404296875 0.6142578125 0.404296875 0.6728515625 0.4189453125 0.701171875 0.4423828125 0.708984375 0.4736328125 0.70703125


================================================
FILE: TumorDetection/train/labels/meningioma_624_jpg.rf.cbc39b6ccc13403b879e6bd5da8ac1c8.txt
================================================
2 0.263671875 0.4580078125 0.263671875 0.4736328125 0.2802734375 0.484375 0.32421875 0.4873046875 0.328125 0.4599609375 0.3134765625 0.4375 0.2880859375 0.4375 0.263671875 0.4580078125
2 0.7080078125 0.45703125 0.6650390625 0.4609375 0.6494140625 0.453125 0.626953125 0.4697265625 0.625 0.5224609375 0.638671875 0.5302734375 0.71875 0.4931640625 0.72265625 0.4853515625 0.708984375 0.4716796875 0.7080078125 0.45703125
2 0.5927734375 0.3359375 0.5205078125 0.33203125 0.4560546875 0.353515625 0.42578125 0.3759765625 0.4306640625 0.404296875 0.4736328125 0.400390625 0.4873046875 0.41015625 0.5380859375 0.3828125 0.5771484375 0.37890625 0.6083984375 0.390625 0.6484375 0.4287109375 0.671875 0.4013671875 0.6640625 0.3798828125 0.6298828125 0.349609375 0.5927734375 0.3359375


================================================
FILE: TumorDetection/train/labels/meningioma_62_jpg.rf.23ac77d0e5264c24c31a81aaffe72da1.txt
================================================
2 0.5107421875 0.583984375 0.4384765625 0.572265625 0.4130859375 0.5859375 0.392578125 0.6083984375 0.396484375 0.6611328125 0.4228515625 0.712890625 0.4765625 0.7275390625 0.525390625 0.6845703125 0.54296875 0.6416015625 0.541015625 0.6123046875 0.5107421875 0.583984375


================================================
FILE: TumorDetection/train/labels/meningioma_638_jpg.rf.63183096ef023fd6179343f832e5b918.txt
================================================
2 0.5302734375 0.1015625 0.5029296875 0.12109375 0.46875 0.1201171875 0.4833984375 0.138671875 0.5068359375 0.138671875 0.5341796875 0.177734375 0.56640625 0.1806640625 0.58203125 0.1689453125 0.58984375 0.1279296875 0.5751953125 0.111328125 0.5302734375 0.1015625


================================================
FILE: TumorDetection/train/labels/meningioma_63_jpg.rf.09d8adece431eaa739db4cd3fa47f681.txt
================================================
2 0.5087890625 0.71484375 0.5390625 0.6376953125 0.537109375 0.6181640625 0.525390625 0.6005859375 0.4482421875 0.568359375 0.4150390625 0.57421875 0.39453125 0.5986328125 0.39453125 0.6513671875 0.408203125 0.6923828125 0.4619140625 0.724609375 0.5087890625 0.71484375


================================================
FILE: TumorDetection/train/labels/meningioma_64_jpg.rf.b86f2a96ab8c4293e2ceb0fdbb133b31.txt
================================================
2 0.51953125 0.6142578125 0.4951171875 0.5859375 0.4404296875 0.57421875 0.41015625 0.5986328125 0.400390625 0.6494140625 0.4072265625 0.662109375 0.4443359375 0.6875 0.458984375 0.6884765625 0.4677734375 0.6796875 0.4990234375 0.685546875 0.51953125 0.6806640625 0.51953125 0.6142578125


================================================
FILE: TumorDetection/train/labels/meningioma_654_jpg.rf.404dce2eaf6f78e861d07817c3b35726.txt
================================================
2 0.4169921875 0.396484375 0.4453125 0.3779296875 0.470703125 0.3369140625 0.478515625 0.2646484375 0.4658203125 0.248046875 0.4462890625 0.240234375 0.3779296875 0.248046875 0.34765625 0.2783203125 0.3359375 0.3134765625 0.3359375 0.3583984375 0.345703125 0.3798828125 0.3740234375 0.3984375 0.4169921875 0.396484375


================================================
FILE: TumorDetection/train/labels/meningioma_656_jpg.rf.8dd8eac691401cc5e8034bcdb6c17947.txt
================================================
2 0.3671875 0.3994140625 0.3798828125 0.419921875 0.4189453125 0.439453125 0.4873046875 0.431640625 0.505859375 0.4208984375 0.529296875 0.3720703125 0.5390625 0.3095703125 0.5126953125 0.201171875 0.4873046875 0.185546875 0.4208984375 0.193359375 0.3427734375 0.244140625 0.328125 0.2646484375 0.31640625 0.3056640625 0.318359375 0.3408203125 0.3359375 0.3740234375 0.3671875 0.3994140625


================================================
FILE: TumorDetection/train/labels/meningioma_657_jpg.rf.f7be9f7383153fc39c77db5d4cc6d494.txt
================================================
2 0.548828125 0.3779296875 0.548828125 0.3505859375 0.533203125 0.3193359375 0.541015625 0.2763671875 0.50390625 0.2255859375 0.501953125 0.1982421875 0.4677734375 0.181640625 0.4208984375 0.189453125 0.3408203125 0.23046875 0.318359375 0.2548828125 0.3203125 0.3544921875 0.33203125 0.3876953125 0.3994140625 0.439453125 0.447265625 0.4443359375 0.4873046875 0.4375 0.5185546875 0.421875 0.537109375 0.4052734375 0.548828125 0.3779296875


================================================
FILE: TumorDetection/train/labels/meningioma_662_jpg.rf.d4f24820353a5f90136f3af5fc78abe1.txt
================================================
2 0.4326171875 0.3828125 0.44921875 0.3896484375 0.4755859375 0.4453125 0.5087890625 0.458984375 0.5380859375 0.4609375 0.5498046875 0.474609375 0.5634765625 0.4765625 0.6064453125 0.466796875 0.654296875 0.4404296875 0.6640625 0.4130859375 0.666015625 0.3603515625 0.66015625 0.3310546875 0.650390625 0.3212890625 0.654296875 0.2939453125 0.6474609375 0.287109375 0.5791015625 0.265625 0.5634765625 0.24609375 0.5224609375 0.236328125 0.4580078125 0.244140625 0.4296875 0.2783203125 0.4296875 0.3056640625 0.416015625 0.3271484375 0.41796875 0.3642578125 0.4326171875 0.3828125


================================================
FILE: TumorDetection/train/labels/meningioma_664_jpg.rf.ff9460f4f6c62598a5aee99e4d8c37fb.txt
================================================
2 0.5439453125 0.431640625 0.5634765625 0.44921875 0.5810546875 0.4453125 0.623046875 0.3857421875 0.625 0.3544921875 0.5888671875 0.32421875 0.5283203125 0.30859375 0.4443359375 0.31640625 0.3798828125 0.34375 0.337890625 0.3779296875 0.3046875 0.4287109375 0.306640625 0.4775390625 0.3212890625 0.494140625 0.3515625 0.5009765625 0.380859375 0.4521484375 0.4072265625 0.4375 0.5126953125 0.423828125 0.5439453125 0.431640625


================================================
FILE: TumorDetection/train/labels/meningioma_677_jpg.rf.533e48c81a54f8bb29c2015e3b47fdad.txt
================================================
2 0.51953125 0.6220703125 0.515625 0.6025390625 0.4912109375 0.578125 0.4580078125 0.578125 0.4365234375 0.552734375 0.3935546875 0.54296875 0.3681640625 0.521484375 0.3388671875 0.521484375 0.3232421875 0.529296875 0.3125 0.5537109375 0.345703125 0.5712890625 0.333984375 0.5947265625 0.333984375 0.6142578125 0.34765625 0.6552734375 0.44921875 0.6923828125 0.4951171875 0.68359375 0.5419921875 0.6484375 0.54296875 0.6396484375 0.51953125 0.6220703125


================================================
FILE: TumorDetection/train/labels/meningioma_689_jpg.rf.f4110e21ea8ba8fdd8b2a45a6127a7c6.txt
================================================
2 0.7041015625 0.306640625 0.6806640625 0.29296875 0.6416015625 0.29296875 0.6279296875 0.2734375 0.5859375 0.2802734375 0.55078125 0.3193359375 0.5390625 0.3544921875 0.55078125 0.3798828125 0.583984375 0.4013671875 0.58203125 0.4130859375 0.599609375 0.4150390625 0.6416015625 0.388671875 0.7109375 0.3662109375 0.716796875 0.3408203125 0.7041015625 0.306640625


================================================
FILE: TumorDetection/train/labels/meningioma_693_jpg.rf.8d77500198a65eae38bcf876f27db18a.txt
================================================
2 0.3896484375 0.3359375 0.416015625 0.3193359375 0.42578125 0.2861328125 0.3974609375 0.244140625 0.3388671875 0.267578125 0.333984375 0.2998046875 0.3447265625 0.3203125 0.3662109375 0.333984375 0.3896484375 0.3359375


================================================
FILE: TumorDetection/train/labels/meningioma_695_jpg.rf.e887b2b087c43c663ead6aeb47cdd2cd.txt
================================================
2 0.482421875 0.2998046875 0.4580078125 0.27734375 0.4013671875 0.265625 0.376953125 0.2841796875 0.3671875 0.3037109375 0.365234375 0.3662109375 0.3857421875 0.392578125 0.4140625 0.3955078125 0.4365234375 0.384765625 0.4609375 0.3583984375 0.48046875 0.3212890625 0.482421875 0.2998046875


================================================
FILE: TumorDetection/train/labels/meningioma_707_jpg.rf.45ba1e9cd6b073aacc11936ced9d3564.txt
================================================
2 0.4287109375 0.396484375 0.5048828125 0.392578125 0.548828125 0.3447265625 0.556640625 0.3115234375 0.552734375 0.2744140625 0.51171875 0.2138671875 0.501953125 0.1865234375 0.4677734375 0.16015625 0.3759765625 0.1796875 0.3134765625 0.212890625 0.294921875 0.2353515625 0.294921875 0.2978515625 0.318359375 0.3173828125 0.3310546875 0.33984375 0.4287109375 0.396484375


================================================
FILE: TumorDetection/train/labels/meningioma_710_jpg.rf.1bba5ad4daee6917d9520f832a0986db.txt
================================================
2 0.3076171875 0.404296875 0.3447265625 0.421875 0.3662109375 0.421875 0.4052734375 0.400390625 0.4912109375 0.37890625 0.5302734375 0.359375 0.544921875 0.3388671875 0.541015625 0.2998046875 0.552734375 0.2880859375 0.5546875 0.2626953125 0.5185546875 0.2109375 0.4169921875 0.220703125 0.3212890625 0.2578125 0.251953125 0.3115234375 0.25390625 0.3310546875 0.279296875 0.3505859375 0.28125 0.3740234375 0.3076171875 0.404296875


================================================
FILE: TumorDetection/train/labels/meningioma_711_jpg.rf.924f1f4a9927e0df7cdb4c153f1807a3.txt
================================================
2 0.52734375 0.3408203125 0.52734375 0.2939453125 0.51171875 0.2763671875 0.505859375 0.2548828125 0.5166015625 0.228515625 0.4580078125 0.248046875 0.3544921875 0.267578125 0.310546875 0.2998046875 0.2890625 0.3408203125 0.298828125 0.3818359375 0.3271484375 0.419921875 0.359375 0.4287109375 0.3818359375 0.41015625 0.4365234375 0.392578125 0.4970703125 0.3515625 0.52734375 0.3408203125


================================================
FILE: TumorDetection/train/labels/meningioma_733_jpg.rf.3df47e74401e0b8871f177d0087ceef2.txt
================================================
2 0.5751953125 0.294921875 0.5478515625 0.291015625 0.52734375 0.3056640625 0.52734375 0.3369140625 0.5419921875 0.3671875 0.568359375 0.3720703125 0.5830078125 0.3671875 0.595703125 0.3505859375 0.6015625 0.3173828125 0.5751953125 0.294921875


================================================
FILE: TumorDetection/train/labels/meningioma_734_jpg.rf.79af18bc94491cc52fb5a9aefdf4b086.txt
================================================
2 0.5888671875 0.39453125 0.5263671875 0.39453125 0.486328125 0.4287109375 0.48828125 0.4912109375 0.50390625 0.5146484375 0.5244140625 0.52734375 0.576171875 0.5283203125 0.61328125 0.4970703125 0.625 0.4775390625 0.615234375 0.4169921875 0.5888671875 0.39453125


================================================
FILE: TumorDetection/train/labels/meningioma_73_jpg.rf.737b7d02c5566539f8c6e99c61f1bb42.txt
================================================
2 0.4921875 0.5673828125 0.474609375 0.5107421875 0.4560546875 0.490234375 0.3603515625 0.45703125 0.3232421875 0.46875 0.296875 0.4931640625 0.263671875 0.5439453125 0.26171875 0.5888671875 0.244140625 0.6123046875 0.263671875 0.6416015625 0.26953125 0.6708984375 0.2919921875 0.69921875 0.3251953125 0.7265625 0.380859375 0.7314453125 0.4306640625 0.720703125 0.458984375 0.6943359375 0.4921875 0.6259765625 0.4921875 0.5673828125


================================================
FILE: TumorDetection/train/labels/meningioma_748_jpg.rf.589414d2dd21bdf2b0752f6fa1959865.txt
================================================
2 0.50390625 0.4228515625 0.509765625 0.4033203125 0.49609375 0.3544921875 0.4755859375 0.330078125 0.4462890625 0.314453125 0.4111328125 0.33203125 0.3955078125 0.357421875 0.3544921875 0.361328125 0.3466796875 0.376953125 0.3154296875 0.38671875 0.296875 0.4111328125 0.3388671875 0.48828125 0.38671875 0.4931640625 0.4287109375 0.490234375 0.4599609375 0.478515625 0.494140625 0.4443359375 0.50390625 0.4228515625
2 0.494140625 0.3525390625 0.4501953125 0.31640625 0.4365234375 0.31640625 0.4072265625 0.337890625 0.400390625 0.3623046875 0.3671875 0.3818359375 0.37890625 0.3935546875 0.37890625 0.4208984375 0.3740234375 0.416015625 0.365234375 0.4365234375 0.353515625 0.4423828125 0.359375 0.4462890625 0.359375 0.4755859375 0.341796875 0.4853515625 0.3564453125 0.4921875 0.412109375 0.4912109375 0.4638671875 0.474609375 0.4921875 0.4462890625 0.5078125 0.4091796875 0.494140625 0.3525390625


================================================
FILE: TumorDetection/train/labels/meningioma_749_jpg.rf.deaf1e32a98b5b9178da63bfdfd3adcd.txt
================================================
2 0.53515625 0.3935546875 0.54296875 0.3759765625 0.54296875 0.3330078125 0.5322265625 0.3203125 0.4892578125 0.310546875 0.4736328125 0.314453125 0.451171875 0.3369140625 0.44921875 0.3837890625 0.4716796875 0.41015625 0.4990234375 0.416015625 0.53515625 0.3935546875


================================================
FILE: TumorDetection/train/labels/meningioma_756_jpg.rf.d642735914e417b8a9c80818bd8cd2e1.txt
================================================
2 0.7380022328125 0.302734375 0.720703125 0.3154296875 0.6884765625 0.30078125 0.6494140625 0.3046875 0.6103515625 0.32421875 0.59375 0.3779296875 0.599609375 0.3974609375 0.6279296875 0.4296875 0.6572265625 0.44140625 0.6796875 0.4404296875 0.6923828125 0.4375 0.72265625 0.3864397328125 0.7896205359375 0.39760044687499996 0.7380022328125 0.302734375


================================================
FILE: TumorDetection/train/labels/meningioma_757_jpg.rf.ba3c9be419657e1322b9c5643a89b7cf.txt
================================================
2 0.787109375 0.3994140625 0.755859375 0.3212890625 0.7255859375 0.294921875 0.6806640625 0.296875 0.6240234375 0.3125 0.59375 0.3544921875 0.607421875 0.4091796875 0.6494140625 0.4453125 0.66796875 0.4462890625 0.6962890625 0.431640625 0.7138671875 0.41015625 0.7568359375 0.41015625 0.787109375 0.3994140625


================================================
FILE: TumorDetection/train/labels/meningioma_758_jpg.rf.773269389e2b3a0c0ed1bfb45ef85194.txt
================================================
2 0.775390625 0.3916015625 0.765625 0.3505859375 0.744140625 0.3134765625 0.7138671875 0.291015625 0.6884765625 0.28515625 0.6640625 0.3037109375 0.6533203125 0.32421875 0.6259765625 0.322265625 0.615234375 0.3291015625 0.60546875 0.3701171875 0.6240234375 0.41015625 0.701171875 0.4208984375 0.7138671875 0.412109375 0.7431640625 0.416015625 0.7685546875 0.408203125 0.775390625 0.3916015625


================================================
FILE: TumorDetection/train/labels/meningioma_759_jpg.rf.b186884d401c9c2a145b3edbdb9cbc6b.txt
================================================
2 0.67265625 0.621875 0.2453125 0.20234375
2 0.7158203125 0.53125 0.701171875 0.5517578125 0.705078125 0.5654296875 0.7412109375 0.58984375 0.765625 0.5947265625 0.771484375 0.5693359375 0.759765625 0.5400390625 0.7431640625 0.529296875 0.7158203125 0.53125
2 0.6630859375 0.5546875 0.6435546875 0.55078125 0.56640625 0.5966796875 0.55859375 0.6103515625 0.560546875 0.6650390625 0.5693359375 0.677734375 0.5927734375 0.68359375 0.6220703125 0.708984375 0.65625 0.7138671875 0.718086621875 0.6779350984375 0.7561383921875 0.654296875 0.78125 0.6096540171875 0.7059151781249999 0.5691964281249999 0.6947544640625 0.5594308031249999 0.6630859375 0.5546875


================================================
FILE: TumorDetection/train/labels/meningioma_773_jpg.rf.c5ae9781857e9a783562dae0412f873b.txt
================================================
2 0.693359375 0.2041015625 0.6298828125 0.138671875 0.6083984375 0.12109375 0.5908203125 0.1171875 0.54296875 0.1533203125 0.54296875 0.2294921875 0.552734375 0.2490234375 0.5810546875 0.271484375 0.630859375 0.2744140625 0.6484375 0.2626953125 0.6748046875 0.22265625 0.689453125 0.2177734375 0.693359375 0.2041015625


================================================
FILE: TumorDetection/train/labels/meningioma_776_jpg.rf.1987625f340e41a7ef8d410a4c5345bd.txt
================================================
2 0.7783203125 0.486328125 0.7490234375 0.486328125 0.7333984375 0.501953125 0.7099609375 0.5078125 0.673828125 0.5517578125 0.673828125 0.5634765625 0.685546875 0.5712890625 0.69140625 0.6142578125 0.7021484375 0.626953125 0.71875 0.6298828125 0.7333984375 0.62890625 0.7578125 0.5908203125 0.783203125 0.5224609375 0.7783203125 0.486328125


================================================
FILE: TumorDetection/train/labels/meningioma_779_jpg.rf.1e1082dfcadaedbd005b5cfc6e06cef2.txt
================================================
2 0.79296875 0.5498046875 0.8134765625 0.525390625 0.828125 0.5224609375 0.822265625 0.4990234375 0.7822265625 0.474609375 0.7158203125 0.498046875 0.6865234375 0.5 0.66015625 0.5263671875 0.662109375 0.5576171875 0.69140625 0.5966796875 0.693359375 0.6318359375 0.7119140625 0.64453125 0.740234375 0.6455078125 0.75390625 0.6318359375 0.763671875 0.6025390625 0.783203125 0.5869140625 0.79296875 0.5498046875


================================================
FILE: TumorDetection/train/labels/meningioma_781_jpg.rf.9d44a2d15a217600ca07c83785ab7d78.txt
================================================
2 0.705078125 0.2431640625 0.71484375 0.2216796875 0.708984375 0.2041015625 0.6865234375 0.1796875 0.6533203125 0.166015625 0.6044921875 0.158203125 0.5009765625 0.1640625 0.4765625 0.2197265625 0.484375 0.2607421875 0.5009765625 0.279296875 0.5126953125 0.279296875 0.5361328125 0.306640625 0.5810546875 0.32421875 0.611328125 0.3232421875 0.646484375 0.3037109375 0.65625 0.2744140625 0.6689453125 0.26171875 0.705078125 0.2431640625


================================================
FILE: TumorDetection/train/labels/meningioma_782_jpg.rf.f20f3818a42dad9ea5ba2054824bfca2.txt
================================================
2 0.3212890625 0.34765625 0.2822265625 0.3515625 0.26171875 0.4052734375 0.279296875 0.4443359375 0.2919921875 0.455078125 0.3662109375 0.46875 0.3837890625 0.490234375 0.40234375 0.4892578125 0.41796875 0.4736328125 0.427734375 0.4365234375 0.408203125 0.4091796875 0.3212890625 0.34765625


================================================
FILE: TumorDetection/train/labels/meningioma_787_jpg.rf.b4fc7fd76dd101524b5e2c01bc4a0dfc.txt
================================================
2 0.7265625 0.5458984375 0.71875 0.5283203125 0.6728515625 0.482421875 0.6455078125 0.48046875 0.5947265625 0.494140625 0.5791015625 0.51171875 0.5654296875 0.51171875 0.5390625 0.5498046875 0.53125 0.6376953125 0.548828125 0.6748046875 0.5595703125 0.6875 0.5849609375 0.689453125 0.5927734375 0.701171875 0.625 0.7060546875 0.681640625 0.6650390625 0.71875 0.6142578125 0.728515625 0.5888671875 0.7265625 0.5458984375


================================================
FILE: TumorDetection/train/labels/meningioma_792_jpg.rf.5cd51b220e3a65606c72bd389a656e38.txt
================================================
2 0.46484375 0.3876953125 0.470703125 0.3662109375 0.46484375 0.3466796875 0.4404296875 0.3203125 0.4072265625 0.31640625 0.40234375 0.3369140625 0.412109375 0.3466796875 0.40234375 0.3583984375 0.4208984375 0.359375 0.419921875 0.3740234375 0.4345703125 0.376953125 0.4365234375 0.384765625 0.4150390625 0.384765625 0.4111328125 0.375 0.4072265625 0.384765625 0.3818359375 0.3828125 0.3740234375 0.3671875 0.3662109375 0.373046875 0.3369140625 0.34375 0.3134765625 0.333984375 0.2880859375 0.3359375 0.263671875 0.3544921875 0.26171875 0.3974609375 0.28515625 0.4541015625 0.3037109375 0.474609375 0.3349609375 0.494140625 0.3662109375 0.498046875 0.3818359375 0.490234375 0.4013671875 0.505859375 0.4296875 0.5068359375 0.47265625 0.4736328125 0.47265625 0.4345703125 0.484375 0.4169921875 0.46484375 0.3876953125


================================================
FILE: TumorDetection/train/labels/meningioma_793_jpg.rf.2b1cddb8721fff606bf90a695ccdf926.txt
================================================
2 0.5087890625 0.54296875 0.4521484375 0.53515625 0.4267578125 0.54296875 0.4013671875 0.5625 0.388671875 0.5986328125 0.404296875 0.6357421875 0.443359375 0.6806640625 0.466796875 0.6845703125 0.501953125 0.6591796875 0.53125 0.6142578125 0.533203125 0.5693359375 0.5087890625 0.54296875


================================================
FILE: TumorDetection/train/labels/meningioma_794_jpg.rf.e6ea105c447a297c2c57171da676b416.txt
================================================
2 0.4326171875 0.34765625 0.408203125 0.3720703125 0.400390625 0.3994140625 0.4326171875 0.44140625 0.46484375 0.4462890625 0.5 0.4365234375 0.501953125 0.3720703125 0.4716796875 0.345703125 0.4326171875 0.34765625
2 0.4306640625 0.34765625 0.408203125 0.3720703125 0.40234375 0.4052734375 0.4326171875 0.44140625 0.46484375 0.4462890625 0.5 0.4365234375 0.501953125 0.3720703125 0.4716796875 0.345703125 0.4306640625 0.34765625


================================================
FILE: TumorDetection/train/labels/meningioma_797_jpg.rf.23361b2204143a425ec4e7e6404ce7e2.txt
================================================
2 0.669921875 0.6943359375 0.6767578125 0.697265625 0.703125 0.6611328125 0.705078125 0.6357421875 0.728515625 0.5810546875 0.73046875 0.5419921875 0.7119140625 0.515625 0.6787109375 0.51171875 0.6435546875 0.529296875 0.625 0.5478515625 0.623046875 0.5732421875 0.609375 0.5986328125 0.66015625 0.6357421875 0.669921875 0.6943359375


================================================
FILE: TumorDetection/train/labels/meningioma_800_jpg.rf.4f2b2bb0491e2960a62f39eb8778e6c0.txt
================================================
2 0.462890625 0.6025390625 0.44140625 0.5712890625 0.3994140625 0.556640625 0.3525390625 0.591796875 0.2958984375 0.60546875 0.248046875 0.6572265625 0.2421875 0.6748046875 0.24609375 0.6943359375 0.2998046875 0.7578125 0.4140625 0.7724609375 0.431640625 0.7626953125 0.46484375 0.6982421875 0.47265625 0.6455078125 0.462890625 0.6025390625


================================================
FILE: TumorDetection/train/labels/meningioma_801_jpg.rf.30da0e61ac1939f9b4fe7cf248e7c52a.txt
================================================
2 0.458984375 0.6064453125 0.4345703125 0.57421875 0.4013671875 0.5625 0.3701171875 0.56640625 0.3544921875 0.58203125 0.3232421875 0.580078125 0.2734375 0.6181640625 0.259765625 0.6552734375 0.26171875 0.6728515625 0.3193359375 0.73828125 0.3818359375 0.765625 0.4140625 0.7705078125 0.4521484375 0.75 0.4609375 0.7333984375 0.466796875 0.6533203125 0.458984375 0.6064453125


================================================
FILE: TumorDetection/train/labels/meningioma_817_jpg.rf.e627f4d8fe10890aa4f608d3811e4590.txt
================================================
2 0.6533203125 0.37109375 0.6123046875 0.357421875 0.59375 0.3662109375 0.595703125 0.3779296875 0.5810546875 0.396484375 0.4912109375 0.384765625 0.466796875 0.3916015625 0.46875 0.4208984375 0.453125 0.4404296875 0.4560546875 0.447265625 0.5263671875 0.44921875 0.55859375 0.4892578125 0.5634765625 0.5078125 0.595703125 0.5126953125 0.63671875 0.4736328125 0.6328125 0.4482421875 0.662109375 0.4072265625 0.6533203125 0.37109375


================================================
FILE: TumorDetection/train/labels/meningioma_818_jpg.rf.d0a0db73578541a04e423e37c4d5cd79.txt
================================================
2 0.5126953125 0.37890625 0.525390625 0.3798828125 0.521484375 0.2744140625 0.5107421875 0.24609375 0.4755859375 0.2421875 0.4638671875 0.244140625 0.4658203125 0.251953125 0.4384765625 0.2578125 0.400390625 0.3154296875 0.3984375 0.3681640625 0.4208984375 0.390625 0.4736328125 0.404296875 0.5126953125 0.37890625


================================================
FILE: TumorDetection/train/labels/meningioma_826_jpg.rf.076b51e8b7ba356f11c7b349609c5525.txt
================================================
2 0.5166015625 0.45703125 0.544921875 0.4384765625 0.544921875 0.4150390625 0.560546875 0.3857421875 0.564453125 0.3525390625 0.5458984375 0.31640625 0.4931640625 0.294921875 0.4794921875 0.279296875 0.4638671875 0.3046875 0.361328125 0.3720703125 0.359375 0.3935546875 0.3759765625 0.427734375 0.4365234375 0.44921875 0.5166015625 0.45703125


================================================
FILE: TumorDetection/train/labels/meningioma_827_jpg.rf.670af4536eb3590bcdb1da7671ee42ca.txt
================================================
2 0.564453125 0.3544921875 0.5439453125 0.3203125 0.4951171875 0.302734375 0.4814453125 0.283203125 0.4462890625 0.3359375 0.4150390625 0.33984375 0.380859375 0.3681640625 0.375 0.4013671875 0.3955078125 0.423828125 0.4384765625 0.443359375 0.5078125 0.4521484375 0.525390625 0.4462890625 0.5390625 0.4111328125 0.560546875 0.3916015625 0.564453125 0.3544921875


================================================
FILE: TumorDetection/train/labels/meningioma_842_jpg.rf.00a12ad362ae2288f7cde416c23d8398.txt
================================================
2 0.5146484375 0.50390625 0.5595703125 0.49609375 0.60546875 0.4560546875 0.6171875 0.4365234375 0.619140625 0.4208984375 0.609375 0.3974609375 0.5537109375 0.353515625 0.5283203125 0.359375 0.498046875 0.3974609375 0.490234375 0.4189453125 0.4921875 0.4619140625 0.501953125 0.4951171875 0.5146484375 0.50390625


================================================
FILE: TumorDetection/train/labels/meningioma_851_jpg.rf.5cb883da6643e552e4b05b22f1f665de.txt
================================================
2 0.6025390625 0.380859375 0.5810546875 0.365234375 0.5693359375 0.365234375 0.5517578125 0.3359375 0.5068359375 0.333984375 0.486328125 0.3662109375 0.47265625 0.4111328125 0.474609375 0.4287109375 0.501953125 0.4794921875 0.498046875 0.4970703125 0.5087890625 0.5078125 0.537109375 0.5087890625 0.5654296875 0.501953125 0.62109375 0.4560546875 0.62890625 0.4130859375 0.6025390625 0.380859375


================================================
FILE: TumorDetection/train/labels/meningioma_864_jpg.rf.dd99018be6b80c0479ba358d9a1e6987.txt
================================================
2 0.3193359375 0.4921875 0.345703125 0.4404296875 0.337890625 0.3916015625 0.3095703125 0.373046875 0.2685546875 0.37109375 0.2548828125 0.375 0.240234375 0.3955078125 0.23828125 0.4560546875 0.248046875 0.4775390625 0.2841796875 0.5 0.3193359375 0.4921875


================================================
FILE: TumorDetection/train/labels/meningioma_866_jpg.rf.6eeaba9ea919ac847aacf7795c4bb3d7.txt
================================================
2 0.359375 0.3095703125 0.3427734375 0.294921875 0.3134765625 0.28515625 0.26953125 0.3232421875 0.2578125 0.3447265625 0.2421875 0.4619140625 0.2587890625 0.48046875 0.322265625 0.4833984375 0.37109375 0.4482421875 0.3828125 0.3994140625 0.376953125 0.3525390625 0.359375 0.3095703125


================================================
FILE: TumorDetection/train/labels/meningioma_867_jpg.rf.c0d07726c11b7a84a5fa4d1fc0a8505f.txt
================================================
2 0.3125 0.41484375 0.17265625 0.19375
2 0.375 0.3916015625 0.36328125 0.3486328125 0.3154296875 0.32421875 0.2958984375 0.326171875 0.267578125 0.3486328125 0.2421875 0.4599609375 0.2431640625 0.474609375 0.29296875 0.4794921875 0.3408203125 0.470703125 0.36328125 0.4521484375 0.373046875 0.4267578125 0.375 0.3916015625


================================================
FILE: TumorDetection/train/labels/meningioma_86_jpg.rf.bdd9bf4edaeff7e33264907091feeccd.txt
================================================
2 0.578125 0.3232421875 0.5498046875 0.30078125 0.5263671875 0.30078125 0.5068359375 0.27734375 0.447265625 0.3310546875 0.439453125 0.3623046875 0.455078125 0.4228515625 0.4853515625 0.443359375 0.517578125 0.4521484375 0.5830078125 0.416015625 0.591796875 0.4072265625 0.59375 0.3798828125 0.578125 0.3232421875


================================================
FILE: TumorDetection/train/labels/meningioma_872_jpg.rf.bcff08f393b9f39298578cc9457afffd.txt
================================================
2 0.3310546875 0.59765625 0.310546875 0.6142578125 0.298828125 0.6494140625 0.3251953125 0.681640625 0.3515625 0.6884765625 0.37890625 0.6630859375 0.38671875 0.6376953125 0.3662109375 0.60546875 0.3310546875 0.59765625


================================================
FILE: TumorDetection/train/labels/meningioma_882_jpg.rf.62b9c653266173b287c6db075d91cba7.txt
================================================
2 0.4482421875 0.365234375 0.48828125 0.3212890625 0.4921875 0.2626953125 0.4580078125 0.212890625 0.3974609375 0.2109375 0.373046875 0.2275390625 0.3515625 0.2666015625 0.3515625 0.2919921875 0.365234375 0.3291015625 0.4111328125 0.3671875 0.4482421875 0.365234375


================================================
FILE: TumorDetection/train/labels/meningioma_889_jpg.rf.08457415fb8ec9d28c77b345a9bd7d65.txt
================================================
2 0.6708984375 0.33984375 0.677734375 0.3251953125 0.677734375 0.2744140625 0.6494140625 0.232421875 0.5283203125 0.177734375 0.5126953125 0.1796875 0.501953125 0.1943359375 0.4765625 0.2607421875 0.427734375 0.3291015625 0.419921875 0.3603515625 0.421875 0.3857421875 0.4365234375 0.408203125 0.4833984375 0.43359375 0.51171875 0.4326171875 0.5673828125 0.4140625 0.6162109375 0.361328125 0.6416015625 0.359375 0.6708984375 0.33984375


================================================
FILE: TumorDetection/train/labels/meningioma_893_jpg.rf.ad34383d7ff8913bb6e72a1927545be6.txt
================================================
2 0.5478515625 0.4140625 0.5966796875 0.37109375 0.62109375 0.2998046875 0.61328125 0.2333984375 0.5859375 0.1904296875 0.5634765625 0.173828125 0.5205078125 0.162109375 0.501953125 0.1806640625 0.4375 0.3486328125 0.4453125 0.4072265625 0.4794921875 0.4296875 0.5478515625 0.4140625


================================================
FILE: TumorDetection/train/labels/meningioma_896_jpg.rf.1088dce4543b6291543f15f3c9a5ef53.txt
================================================
2 0.357421875 0.2568359375 0.3212890625 0.228515625 0.2607421875 0.23046875 0.2099609375 0.248046875 0.1708984375 0.26953125 0.1328125 0.3212890625 0.125 0.3759765625 0.13671875 0.4248046875 0.1591796875 0.451171875 0.1689453125 0.453125 0.1845703125 0.451171875 0.189453125 0.4287109375 0.1982421875 0.4296875 0.220703125 0.4990234375 0.228515625 0.5048828125 0.2783203125 0.4921875 0.314453125 0.4560546875 0.345703125 0.3974609375 0.357421875 0.2568359375


================================================
FILE: TumorDetection/train/labels/meningioma_901_jpg.rf.7f26f443a85ace9eeb2e372adf5b737d.txt
================================================
2 0.2412109375 0.490234375 0.2978515625 0.45703125 0.31640625 0.4287109375 0.32421875 0.3701171875 0.30859375 0.3232421875 0.2880859375 0.30859375 0.2529296875 0.32421875 0.236328125 0.3505859375 0.228515625 0.3857421875 0.2265625 0.4384765625 0.234375 0.4580078125 0.232421875 0.4833984375 0.2412109375 0.490234375


================================================
FILE: TumorDetection/train/labels/meningioma_913_jpg.rf.8fa76a39c193b325ce78fe62c88ce2c2.txt
================================================
2 0.6328125 0.5126953125 0.5986328125 0.494140625 0.5673828125 0.498046875 0.546875 0.5166015625 0.56640625 0.5595703125 0.59765625 0.5810546875 0.6083984375 0.603515625 0.638671875 0.6083984375 0.646484375 0.6064453125 0.634765625 0.5927734375 0.64453125 0.5380859375 0.6328125 0.5126953125


================================================
FILE: TumorDetection/train/labels/meningioma_91_jpg.rf.9376f3bcef6ca61b2d4f0dde0d35a9b0.txt
================================================
2 0.6748046875 0.69140625 0.7119140625 0.689453125 0.712890625 0.6806640625 0.6728515625 0.66015625 0.6318359375 0.6015625 0.5810546875 0.58984375 0.5595703125 0.59375 0.537109375 0.6162109375 0.533203125 0.6728515625 0.5751953125 0.712890625 0.6015625 0.7255859375 0.6171875 0.7021484375 0.611328125 0.6767578125 0.6298828125 0.66015625 0.6748046875 0.69140625


================================================
FILE: TumorDetection/train/labels/meningioma_924_jpg.rf.ef23b2a126ea5c99ead25e46dcbb67e6.txt
================================================
2 0.5947265625 0.263671875 0.5458984375 0.248046875 0.5224609375 0.25 0.5078125 0.2626953125 0.490234375 0.2822265625 0.486328125 0.3115234375 0.498046875 0.3603515625 0.5126953125 0.37109375 0.537109375 0.3720703125 0.568359375 0.3583984375 0.607421875 0.2958984375 0.60546875 0.2763671875 0.5947265625 0.263671875


================================================
FILE: TumorDetection/train/labels/meningioma_935_jpg.rf.2820d1d475b4a80653760ac515bcf47f.txt
================================================
2 0.2822265625 0.3203125 0.25390625 0.3369140625 0.259765625 0.3662109375 0.25390625 0.4072265625 0.2724609375 0.42578125 0.30078125 0.4306640625 0.333984375 0.4072265625 0.349609375 0.3564453125 0.3330078125 0.33203125 0.2822265625 0.3203125


================================================
FILE: TumorDetection/train/labels/meningioma_937_jpg.rf.8da0c940919adb2db3045845d5a731e6.txt
================================================
2 0.564453125 0.5478515625 0.55078125 0.4892578125 0.5087890625 0.47265625 0.4619140625 0.484375 0.427734375 0.5361328125 0.4326171875 0.55859375 0.4853515625 0.56640625 0.5283203125 0.56640625 0.564453125 0.5478515625


================================================
FILE: TumorDetection/train/labels/meningioma_944_jpg.rf.86b8a982d384dfff1deeaa03e63e97db.txt
================================================
2 0.5498046875 0.181640625 0.5107421875 0.1875 0.4921875 0.2333984375 0.494140625 0.2666015625 0.5263671875 0.29296875 0.609375 0.2939453125 0.62890625 0.2607421875 0.62890625 0.2060546875 0.6083984375 0.193359375 0.5498046875 0.181640625


================================================
FILE: TumorDetection/train/labels/meningioma_958_jpg.rf.c45c2745072466a253c58e32f0360c49.txt
================================================
2 0.6865234375 0.6328125 0.6494140625 0.61328125 0.6240234375 0.611328125 0.603515625 0.6240234375 0.587890625 0.6572265625 0.587890625 0.6826171875 0.59765625 0.6982421875 0.59375 0.7236328125 0.6220703125 0.73828125 0.6640625 0.7412109375 0.70703125 0.7060546875 0.71875 0.6435546875 0.7099609375 0.6328125 0.6865234375 0.6328125


================================================
FILE: TumorDetection/train/labels/meningioma_95_jpg.rf.ed230bc3cdc5214e8911b1958cf547e3.txt
================================================
2 0.4013671875 0.287109375 0.3720703125 0.298828125 0.35546875 0.3173828125 0.3603515625 0.330078125 0.3955078125 0.341796875 0.423828125 0.3408203125 0.451171875 0.2958984375 0.4462890625 0.271484375 0.4306640625 0.287109375 0.4013671875 0.287109375


================================================
FILE: TumorDetection/train/labels/meningioma_962_jpg.rf.dc5529c452baf31a925796b88183ea0b.txt
================================================
2 0.5537109375 0.40625 0.60546875 0.3408203125 0.603515625 0.3056640625 0.583984375 0.2529296875 0.5185546875 0.224609375 0.5009765625 0.22265625 0.482421875 0.2392578125 0.451171875 0.3076171875 0.435546875 0.4033203125 0.4794921875 0.42578125 0.4990234375 0.42578125 0.5537109375 0.40625


================================================
FILE: TumorDetection/train/labels/meningioma_983_jpg.rf.d9b0a28a097dc6b2bb1de1e2a293f3a1.txt
================================================
2 0.5771484375 0.154296875 0.5595703125 0.154296875 0.5244140625 0.169921875 0.498046875 0.1982421875 0.48828125 0.2314453125 0.4931640625 0.263671875 0.5380859375 0.28125 0.564453125 0.2802734375 0.6015625 0.2705078125 0.62109375 0.2119140625 0.619140625 0.1962890625 0.5771484375 0.154296875


================================================
FILE: TumorDetection/train/labels/meningioma_986_jpg.rf.bcacf1010002848f9f3dd0c9dd22fb3d.txt
================================================
2 0.5771484375 0.16015625 0.5009765625 0.16796875 0.482421875 0.2275390625 0.486328125 0.2626953125 0.5361328125 0.302734375 0.546875 0.3037109375 0.5908203125 0.294921875 0.62109375 0.2509765625 0.6171875 0.1904296875 0.5771484375 0.16015625


================================================
FILE: TumorDetection/train/labels/meningioma_989_jpg.rf.d71f7461ce2b6395ac9695c8409cd1b1.txt
================================================
2 0.5693359375 0.498046875 0.5166015625 0.501953125 0.509765625 0.5263671875 0.5234375 0.5458984375 0.521484375 0.5556640625 0.5380859375 0.5703125 0.560546875 0.5712890625 0.5869140625 0.568359375 0.603515625 0.5537109375 0.603515625 0.5146484375 0.5693359375 0.498046875


================================================
FILE: TumorDetection/train/labels/meningioma_992_jpg.rf.7ecd4a2363705b5217df4eb8f7b620e5.txt
================================================
2 0.5302734375 0.61328125 0.5517578125 0.64453125 0.5771484375 0.65234375 0.6103515625 0.646484375 0.642578125 0.6181640625 0.658203125 0.5732421875 0.6533203125 0.509765625 0.6103515625 0.4921875 0.5654296875 0.4921875 0.5107421875 0.505859375 0.484375 0.5283203125 0.482421875 0.5595703125 0.4921875 0.5849609375 0.5087890625 0.60546875 0.5302734375 0.61328125


================================================
FILE: TumorDetection/train/labels/meningioma_993_jpg.rf.4bf82291957b6780725ba736bfbb4ae0.txt
================================================
2 0.47265625 0.4794921875 0.4501953125 0.46484375 0.4208984375 0.46875 0.3759765625 0.501953125 0.3369140625 0.517578125 0.333984375 0.5419921875 0.3203125 0.5537109375 0.3662109375 0.58984375 0.41796875 0.6005859375 0.455078125 0.5810546875 0.486328125 0.5244140625 0.486328125 0.5029296875 0.47265625 0.4794921875


================================================
FILE: TumorDetection/train/labels/meningioma_99_jpg.rf.14fda29cfbc391267067f1a140776f9e.txt
================================================
2 0.431640625 0.5439453125 0.3994140625 0.51953125 0.3623046875 0.529296875 0.333984375 0.5595703125 0.328125 0.6064453125 0.3525390625 0.6328125 0.37109375 0.6357421875 0.4169921875 0.623046875 0.439453125 0.5927734375 0.439453125 0.5595703125 0.431640625 0.5439453125


================================================
FILE: TumorDetection/train/labels/no_tumor_1012_jpg.rf.b7971dea3b79562d4e76cd9c3e6bbad6.txt
================================================
0 0.854968228125 0.8232421875 0.9038235546875001 0.7373046875 0.9453505828125 0.6201171875 0.942907815625 0.4384765640625 0.9062663203125 0.3232421875 0.8427543953125 0.1982421875 0.7987846015625 0.1396484359375 0.7340512921875 0.08984375 0.6485544703125 0.0527343734375 0.54107275 0.033203125 0.41893443125 0.037109373437499996 0.34809420625 0.05078125 0.23328418593750003 0.095703125 0.1807647109375 0.1337890625 0.12702385 0.2138671875 0.063511925 0.3466796875 0.0366414953125 0.4931640625 0.046412559375 0.6708984359375 0.1001534203125 0.7900390625 0.1367949171875 0.8427734359375 0.24061248750000003 0.91796875 0.3212237765625 0.94921875 0.4934388046875 0.9697265640625 0.5825997765625 0.96484375 0.6412261703125 0.94921875 0.8244336484374999 0.85546875 0.854968228125 0.8232421875


================================================
FILE: TumorDetection/train/labels/no_tumor_1021_jpg.rf.61b4d9a8483a617a2cdd62ec51c14fad.txt
================================================
0 0.740234375 0.779579740625 0.763671875 0.732510775 0.80078125 0.6344504296875 0.80859375 0.57757543125 0.80859375 0.49128232812499995 0.79296875 0.38537715625 0.77734375 0.33242456875 0.732421875 0.23828663749999998 0.685546875 0.1657219828125 0.6357421875 0.1176724140625 0.5869140625 0.09021551718750001 0.5244140625 0.0745258625 0.4736328125 0.0725646546875 0.3994140625 0.0823706890625 0.3505859375 0.09806034531249999 0.3076171875 0.125517240625 0.25390625 0.18337284375000001 0.173828125 0.3422306046875 0.15234375 0.457941809375 0.1484375 0.5501185359375 0.158203125 0.6246443968750001 0.173828125 0.679558190625 0.234375 0.8031142234375 0.259765625 0.8364547421875 0.2861328125 0.8590086203125 0.3720703125 0.900193965625 0.4033203125 0.90803879375 0.5078125 0.9129418093749999 0.5751953125 0.9060775859374999 0.6083984375 0.8923491374999999 0.7099609375 0.8119396546875001 0.740234375 0.779579740625


================================================
FILE: TumorDetection/train/labels/no_tumor_1039_jpg.rf.01d362e9945468a352a542a18c38ded1.txt
================================================
0 0.9036847015625 0.7783203125 0.9571284171875 0.6044921875 0.9571284171875 0.4326171875 0.8210898609375 0.1669921875 0.7129877953125 0.06835937656249999 0.5575151593750001 0.019531248437500003 0.365603621875 0.042968751562500004 0.245355253125 0.1123046875 0.1068874375 0.3173828125 0.051014457812500004 0.4755859359375 0.060731496875 0.6455078125 0.1190337359375 0.7880859359375 0.256286925 0.9121093765625 0.3959693734375 0.9541015640625 0.664402596875 0.953125 0.771290034375 0.9140625 0.855099503125 0.8466796875 0.9036847015625 0.7783203125


================================================
FILE: TumorDetection/train/labels/no_tumor_1041_jpg.rf.c2ad0c6c5b3237dcb37ed3bef1ba557e.txt
================================================
0 0.5673828125 0.8515625 0.6650390625 0.814453125 0.755859375 0.7255859375 0.794921875 0.6474609375 0.8125 0.5595703125 0.810546875 0.5029296875 0.798828125 0.4638671875 0.7578125 0.3818359375 0.7539062515625 0.2919921875 0.71875 0.2255859375 0.6611328125 0.171875 0.6396484375 0.1601562515625 0.5166015625 0.142578125 0.4990234375 0.1679687484375 0.4833984375 0.142578125 0.4638671875 0.140625 0.4052734375 0.1484375 0.3330078125 0.1796875 0.291015625 0.2216796875 0.2539062515625 0.2744140625 0.244140625 0.3154296875 0.2421875 0.3857421875 0.1914062515625 0.4931640625 0.1875 0.5458984375 0.1953125 0.5693359375 0.197265625 0.6142578125 0.216796875 0.6572265625 0.2226562515625 0.6884765625 0.2578125 0.7392578125 0.3017578125 0.791015625 0.3681640625 0.8359375 0.4072265625 0.8515625 0.453125 0.8564453125 0.4736328125 0.833984375 0.4814453125 0.8359375 0.4912109375 0.8515625 0.5673828125 0.8515625


================================================
FILE: TumorDetection/train/labels/no_tumor_1045_jpg.rf.2634e2ce15ebbd1c21f43d70e8044f8f.txt
================================================
0 0.79296875 0.70222355625 0.8046875 0.46042491406249997 0.7734375 0.347585546875 0.765625 0.23474618125000002 0.7265625 0.172281534375 0.6767578109375 0.12089932187500001 0.6201171890625 0.09268947968749999 0.4560546890625 0.06447963749999999 0.3388671890625 0.0906744890625 0.2285156234375 0.1964613984375 0.1503906234375 0.406020221875 0.13671875 0.6719987281249999 0.1660156234375 0.7687181859375001 0.2607421890625 0.8966699640625 0.4228515609375 0.9893594484375001 0.48046875 0.990366940625 0.6181640609375 0.9671945703124999 0.71484375 0.8916324937500001 0.7753906234375 0.77476315 0.79296875 0.70222355625


================================================
FILE: TumorDetection/train/labels/no_tumor_1050_jpg.rf.b94da538ccc63eb4220a4679c5b88f84.txt
================================================
0 0.673828125 0.7589432578124999 0.69140625 0.7415761328125 0.705078125 0.6825279171874999 0.70703125 0.626953125 0.697265625 0.6026391531249999 0.7109375 0.547064359375 0.703125 0.4845427203125 0.693359375 0.4671755984375 0.69140625 0.4150742296875 0.677734375 0.39770710781249996 0.6796875 0.3699197125 0.671875 0.3525525875 0.673828125 0.300451221875 0.66015625 0.248349853125 0.6328125 0.2031953359375 0.62890625 0.161514240625 0.6142578125 0.1424104046875 0.5888671875 0.145883828125 0.5859375 0.1059394484375 0.5576171875 0.076415340625 0.5166015625 0.0659950671875 0.505859375 0.0850989 0.5009765625 0.173671225 0.490234375 0.154567390625 0.49609375 0.09204575000000001 0.4853515625 0.076415340625 0.4541015625 0.0972558859375 0.4228515625 0.0937824625 0.41015625 0.109412871875 0.4072265625 0.1701978 0.3935546875 0.1597775296875 0.375 0.16846108906249999 0.373046875 0.2031953359375 0.36328125 0.2379295796875 0.34765625 0.2622435515625 0.33203125 0.3212917671875 0.345703125 0.3525525875 0.33203125 0.3733931359375 0.33203125 0.387286834375 0.349609375 0.4254945046875 0.3330078125 0.42375778906250006 0.322265625 0.43938820156249997 0.3203125 0.49843641875 0.30859375 0.5505377859375 0.30859375 0.5783251828125 0.31640625 0.6026391531249999 0.310546875 0.67558106875 0.35546875 0.8179914734375 0.4013671875 0.8822498281250001 0.4560546875 0.910037225 0.4912109375 0.910037225 0.5029296875 0.8857232515625 0.5126953125 0.8822498281250001 0.5244140625 0.910037225 0.537109375 0.9152473593750001 0.5849609375 0.910037225 0.6064453125 0.8961435234374999 0.671875 0.8006243515625 0.673828125 0.7589432578124999


================================================
FILE: TumorDetection/train/labels/no_tumor_1068_jpg.rf.2d663ba9dee8887033595f92974557c5.txt
================================================
0 0.9925781265625 0.5087890609375 0.9745312531250001 0.3369140609375 0.8610937468750001 0.1533203125 0.7386328140625 0.058593746875 0.6019921859374999 0.011718746875 0.369960940625 0.0253906265625 0.17789062656250001 0.1298828125 0.0721875 0.2763671875 0.0077343734375 0.4638671875 0.012890626562499998 0.7392578125 0.077343746875 0.8330078125 0.163710940625 0.90625 0.2951953140625 0.964843746875 0.3751171859375 0.9863281265625 0.556875 0.9912109390625 0.7798828140625 0.9296875 0.928125 0.8193359390625 0.9745312531250001 0.6806640609375 0.9925781265625 0.5087890609375


================================================
FILE: TumorDetection/train/labels/no_tumor_1073_jpg.rf.5bc35117e317c026e72719bc078cbd8c.txt
================================================
0 0.802734375 0.6259765640625 0.8085937515625 0.4912109359375 0.755859375 0.3798828125 0.75 0.2783203125 0.720703125 0.2275390640625 0.6884765640625 0.1953125 0.6513671875 0.166015625 0.6142578125 0.154296875 0.5380859359375 0.142578125 0.4189453125 0.1445312484375 0.3291015640625 0.1796875 0.291015625 0.2197265640625 0.25 0.2841796875 0.240234375 0.3994140640625 0.193359375 0.4873046875 0.1875 0.5458984359375 0.197265625 0.6162109359375 0.2265625 0.6923828125 0.2578125 0.7392578125 0.3076171875 0.796875 0.3623046875 0.8320312484375 0.3994140640625 0.849609375 0.462890625 0.8564453125 0.5830078125 0.8515625 0.6787109359375 0.8046875 0.7578125 0.7236328125 0.802734375 0.6259765640625


================================================
FILE: TumorDetection/train/labels/no_tumor_1079_jpg.rf.ae610d05451e54ff6492a5e213ebd372.txt
================================================
0 0.8417968765625 0.646015453125 0.8398437484375 0.45999535312499995 0.7890625015625 0.251733721875 0.6767578125 0.1233394140625 0.6318359359375 0.0930100484375 0.5595703109375 0.0768343875 0.4033203109375 0.08492222031249999 0.2646484390625 0.1880420578125 0.2011718765625 0.2962167890625 0.1601562515625 0.48830276250000004 0.162109375 0.686454603125 0.2421874984375 0.8583209999999999 0.3740234390625 0.960429859375 0.484375 0.9836823703125 0.6259765609375 0.96649573125 0.7412109359375 0.8775295984374999 0.7949218765625 0.8017061890624999 0.8417968765625 0.646015453125


================================================
FILE: TumorDetection/train/labels/no_tumor_1085_jpg.rf.c4e0c79ffc9f4a82cbf6a27f594bd1ce.txt
================================================
0 0.9449744609374999 0.7392578125 0.9762995765625 0.5771484390625 0.9501953125 0.3525390609375 0.903207634375 0.2646484390625 0.8431678171874999 0.1904296875 0.7504976734375 0.11914062656249999 0.6147554859375 0.0625 0.40592135156250003 0.06835937343750001 0.2649583078125 0.1328125 0.1566256015625 0.2314453125 0.0991962109375 0.3251953125 0.06265024062499999 0.4326171875 0.057429384375 0.5830078125 0.120079625 0.8154296875 0.332829403125 0.9277343734375 0.519474909375 0.9443359390625 0.6095346296875 0.9394531265625 0.7217829765625 0.9101562515625 0.9084284859375 0.7880859390625 0.9449744609374999 0.7392578125


================================================
FILE: TumorDetection/train/labels/no_tumor_1088_jpg.rf.9a5310703a4cb1600585cfc4dae70d87.txt
================================================
0 0.76806640625 0.8212890625 0.81730143125 0.7158203125 0.8232096343749999 0.6669921875 0.821240234375 0.5224609375 0.8054850265625 0.4169921875 0.79169921875 0.3681640625 0.7582194015625 0.2939453125 0.7011067703125 0.2138671875 0.6508870453125 0.16796875 0.5977132171874999 0.13671875 0.5307535796875 0.12109375 0.45985514375000003 0.12109375 0.396834309375 0.13671875 0.3416910796875 0.166015625 0.3091959640625 0.1982421875 0.263899740625 0.2705078125 0.2126953140625 0.4150390625 0.20481770781250003 0.5205078125 0.206787109375 0.5947265625 0.22254231875000002 0.6923828125 0.25011393125 0.7841796875 0.2796549484375 0.8310546875 0.329874675 0.8828124984375 0.39092610625 0.916015625 0.4362223296875 0.9296875015625 0.467732746875 0.935546875 0.5514322921875 0.9345703125 0.62725423125 0.91796875 0.6902750656250001 0.888671875 0.72178548125 0.87109375 0.76806640625 0.8212890625


================================================
FILE: TumorDetection/train/labels/no_tumor_1093_jpg.rf.aea4c307a9ae60e321de8879b8adae27.txt
================================================
0 0.9670054609375001 0.4677734390625 0.892976334375 0.2607421890625 0.7934996984375 0.11621093906249999 0.7275675046875001 0.07421875 0.62577745625 0.041015623437499996 0.45689851406250004 0.02734375 0.3111536703125 0.05859375 0.1619387140625 0.1669921890625 0.097163228125 0.2646484390625 0.048581610937500005 0.4326171890625 0.0300743296875 0.6083984390625 0.10641686875 0.7568359390625 0.21861726250000002 0.8730468765625 0.36204869375 0.9394531234375 0.5598452671875 0.9599609390625 0.6951797625 0.93359375 0.84670813125 0.8271484390625 0.9577518203125001 0.6083984390625 0.9670054609375001 0.4677734390625


================================================
FILE: TumorDetection/train/labels/no_tumor_1103_jpg.rf.f5d51689f9a518c267c2095c9ee98bf7.txt
================================================
0 0.8320312484375 0.7275390625 0.8378906265625 0.4619140625 0.794921875 0.3134765640625 0.6943359375 0.1796875015625 0.5556640625 0.11914062656249999 0.4150390625 0.1289062484375 0.3095703109375 0.1914062484375 0.2148437515625 0.3427734359375 0.1835937515625 0.4580078140625 0.1914062484375 0.7314453109375 0.2285156265625 0.8330078140625 0.3681640625 0.951171875 0.513671875 0.9833984359375 0.6201171859375 0.966796875 0.6923828140625 0.9257812484375 0.779296875 0.8408203109375 0.8320312484375 0.7275390625


================================================
FILE: TumorDetection/train/labels/no_tumor_1113_jpg.rf.152593fcacfb8267ecd4d1e1e23db8df.txt
================================================
0 0.8949068515624999 0.8095703140625 0.9240642171875001 0.7275390625 0.9307928406249999 0.5927734375 0.910606971875 0.3701171859375 0.8949068515624999 0.2919921859375 0.8724781078125 0.2470703140625 0.8141633765625 0.1533203140625 0.7592129562500001 0.10546875156249999 0.6762266046875001 0.060546875 0.6223976203125 0.042968751562500004 0.514739653125 0.0273437515625 0.43848192812500003 0.0273437515625 0.34203833281249996 0.042968751562500004 0.2657806078125 0.07421875156249999 0.1884014421875 0.1298828140625 0.12335808749999999 0.2119140625 0.076257725 0.3212890625 0.0493432359375 0.5107421859375 0.0493432359375 0.6396484375 0.1031722203125 0.8037109375 0.2321374921875 0.9335937515625 0.3554955796875 0.9882812484375 0.4452105515625 1 0.5562328296875 0.9990234375 0.63585486875 0.9882812484375 0.7300555875 0.947265625 0.8780852937500001 0.8320312484375 0.8949068515624999 0.8095703140625


================================================
FILE: TumorDetection/train/labels/no_tumor_1116_jpg.rf.58d57065c60fae62176fd8277ddfe6c0.txt
================================================
0 0.8301109109375 0.6376953109375 0.844012290625 0.5205078109375 0.822167265625 0.4345703109375 0.788406775 0.3681640609375 0.7804631296875 0.2626953109375 0.7347871703125 0.1845703109375 0.7218787484375 0.171875 0.694075990625 0.166015625 0.6762027859375 0.14453125 0.6146395390625 0.12109375 0.5252735296875 0.123046875 0.49747077187500005 0.11328125 0.435907525 0.111328125 0.34058378437500003 0.1484375 0.2819994 0.2060546890625 0.2760416671875 0.2314453109375 0.25221073125000004 0.2607421890625 0.238309353125 0.3642578109375 0.19263339375 0.4501953109375 0.1846897484375 0.5654296890625 0.206534771875 0.6533203109375 0.23632344062500002 0.7236328109375 0.2988796453125 0.828125 0.4160484125 0.921875 0.46768210625 0.919921875 0.49945668593750003 0.869140625 0.5213017078125 0.921875 0.5679706234374999 0.9345703109375 0.6583295875 0.900390625 0.7159210125000001 0.841796875 0.752660371875 0.8193359390625 0.8301109109375 0.6376953109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1122_jpg.rf.b62a4d7cc6933fd7161d203f8dd61875.txt
================================================
0 0.796875 0.5048828125 0.779296875 0.2783203125 0.7421875 0.2373046875 0.740234375 0.1767578125 0.6640625 0.1181640625 0.6552734375 0.087890625 0.3876953125 0.087890625 0.3037109375 0.140625 0.2578125 0.2021484375 0.1796875 0.4951171875 0.173828125 0.6376953125 0.19921875 0.6689453125 0.21875 0.7822265625 0.2509765625 0.791015625 0.3447265625 0.890625 0.4072265625 0.9140625 0.49609375 0.9150390625 0.6572265625 0.9140625 0.740234375 0.8212890625 0.783203125 0.7041015625 0.796875 0.5048828125


================================================
FILE: TumorDetection/train/labels/no_tumor_112_jpg.rf.afdc41cfb3f5fbb0c626fcff74f2d6e5.txt
================================================
0 0.799804690625 0.1533203140625 0.706494140625 0.08007812343750001 0.5034667953125 0.058593753125 0.312744140625 0.121093753125 0.2266113296875 0.1796875 0.172265625 0.3310546859375 0.2153320296875 0.4521484375 0.3496582046875 0.4511718765625 0.401953125 0.4267578140625 0.400927734375 0.371093753125 0.402978515625 0.386718753125 0.419384765625 0.371093753125 0.4562988296875 0.378906246875 0.4327148453125 0.4013671859375 0.4645019546875 0.417968753125 0.5219238296875 0.3828125 0.589599609375 0.3828125 0.578320309375 0.3623046859375 0.6039550796875 0.3515625 0.67265625 0.6376953140625 0.6952148453125 0.6455078140625 0.829541015625 0.5449218765625 0.9208007796875 0.4306640625 0.914648440625 0.3076171859375 0.799804690625 0.1533203140625


================================================
FILE: TumorDetection/train/labels/no_tumor_1153_jpg.rf.44296b2708ffccf2650bd2dc305fc38f.txt
================================================
0 0.85546875 0.54208846875 0.8623046859375 0.5648107421875 0.8691406265625 0.5247762625 0.86328125 0.3538182203125 0.7988281265625 0.20882849218750002 0.6650390640625 0.064920775 0.5693359359375 0.025968309374999998 0.4912109359375 0.0302963609375 0.4326171859375 0.047608568750000003 0.3544921859375 0.11252934218749999 0.2753906265625 0.2023364125 0.2109375 0.3256858875 0.1933593734375 0.4533634078125 0.2109375 0.7065544312500001 0.23828125 0.7195385875 0.24609375 0.7996075406250001 0.29296875 0.8861685734375 0.3408203140625 0.9240390234375001 0.3505859359375 0.913218896875 0.4013671859375 0.9370231828125 0.4130859359375 0.96948356875 0.6640625 0.974893634375 0.6923828140625 0.913218896875 0.80859375 0.819083775 0.8339843734375 0.7498349484375 0.85546875 0.54208846875


================================================
FILE: TumorDetection/train/labels/no_tumor_1157_jpg.rf.93f41da013b86c9dac838699ebbce758.txt
================================================
0 0.943593746875 0.5224609390625 0.9203906265624999 0.5009765609375 0.9487500000000001 0.4931640609375 0.9152343734375 0.4580078125 0.9410156265625 0.4365234390625 0.9358593734375 0.4013671875 0.902343746875 0.4130859390625 0.788906253125 0.2119140609375 0.7321875 0.1767578125 0.7386328140625 0.15625 0.6664453140625 0.1621093734375 0.669023440625 0.1484375 0.5813671859375 0.12304687343750001 0.483398440625 0.1425781265625 0.39058594062499996 0.125 0.3261328140625 0.1484375 0.3132421859375 0.1308593734375 0.308085940625 0.1484375 0.268125 0.1494140609375 0.2023828140625 0.2246093734375 0.1340625 0.2470703125 0.144375 0.2802734390625 0.1108593734375 0.2880859390625 0.1417968734375 0.2958984390625 0.1108593734375 0.3505859390625 0.0541406265625 0.3544921875 0.041249999999999995 0.4189453125 0.0683203140625 0.441406253125 0.0850781265625 0.4228515609375 0.0902343734375 0.4736328125 0.0721875 0.6845703125 0.1882031265625 0.8408203125 0.27199218593749996 0.9042968734375 0.40089844062499996 0.9433593734375 0.5749218734375 0.9482421875 0.7489453140625 0.9121093734375 0.8868750000000001 0.8095703125 0.9384375 0.7080078125 0.943593746875 0.5224609390625
4 0.4975781265625 0.5830078125 0.515625 0.5751953125 0.516914059375 0.5625 0.53625 0.6142578125 0.5671875 0.6337890609375 0.6045703140625001 0.6796875 0.638085940625 0.6855468734375 0.654843746875 0.6650390609375 0.639375 0.5869140609375 0.6084375 0.5439453125 0.5852343734375001 0.4814453125 0.5710546859375001 0.34375 0.5259375000000001 0.3564453125 0.515625 0.3955078125 0.5040234406249999 0.3984375 0.4975781265625 0.3662109390625 0.475664059375 0.34375 0.41636718593750005 0.3222656265625 0.3944531265625 0.3232421875 0.3944531265625 0.3623046875 0.4253906265625 0.3974609390625 0.4253906265625 0.4501953125 0.4125 0.5029296875 0.350625 0.6259765609375 0.350625 0.6787109390625 0.386718746875 0.6865234390625 0.43054687343750003 0.6708984390625 0.45375 0.6337890609375 0.495 0.6064453125 0.4975781265625 0.5830078125


================================================
FILE: TumorDetection/train/labels/no_tumor_1210_jpg.rf.4ce02806ade4c1f448b354d2e6bc60cc.txt
================================================
0 0.7246093734375 0.3701171859375 0.6572265640625 0.2304687515625 0.5810546890625 0.1875 0.5283203109375 0.1796875015625 0.4990234359375 0.1972656265625 0.4443359375 0.173828125 0.3564453109375 0.2070312484375 0.2890624984375 0.2919921859375 0.2460937515625 0.4814453109375 0.2421875015625 0.5849609375 0.263671875 0.6787109375 0.3212890625 0.7773437515625 0.4169921859375 0.8320312484375 0.4794921859375 0.8183593734375 0.546875 0.8349609375 0.6318359375 0.8007812484375 0.71875 0.6943359375 0.7441406265625 0.5654296890625 0.7246093734375 0.3701171859375


================================================
FILE: TumorDetection/train/labels/no_tumor_1215_jpg.rf.5839678b4f833068a54ee551626d4cac.txt
================================================
0 0.8320312484375 0.7255859375 0.8378906265625 0.4541015640625 0.794921875 0.3095703109375 0.6982421859375 0.1816406265625 0.5595703109375 0.11914062656249999 0.4111328140625 0.1289062484375 0.3095703109375 0.189453125 0.2148437515625 0.3408203109375 0.1835937515625 0.4482421859375 0.189453125 0.7255859375 0.21875 0.8193359375 0.3642578140625 0.951171875 0.4960937515625 0.9853515640625 0.6005859375 0.9746093734375 0.6943359375 0.9257812484375 0.7871093734375 0.8310546890625 0.8320312484375 0.7255859375


================================================
FILE: TumorDetection/train/labels/no_tumor_1237_jpg.rf.2b4e1824fadb8f068b44eaa43a7ea4fe.txt
================================================
0 0.7636718765625 0.6518554671875 0.7636718765625 0.5141601578125 0.7324218765625 0.2797851578125 0.6542968765625 0.1303710921875 0.5869140625 0.0615234359375 0.4775390625 0.038085935937499996 0.3759765625 0.07324218593750001 0.2753906234375 0.2475585921875 0.24609375 0.4877929671875 0.24609375 0.6665039078125 0.2519531234375 0.7250976578125 0.2988281234375 0.8217773421875 0.3623046875 0.919921875 0.4228515625 0.9580078140625 0.53125 0.9682617171875 0.5830078125 0.9580078140625 0.6376953125 0.9228515640625 0.7324218765625 0.7924804671875 0.7636718765625 0.6518554671875


================================================
FILE: TumorDetection/train/labels/no_tumor_1251_jpg.rf.93d7f42c965eb13ed1d52349e5b56fb3.txt
================================================
0 0.9049218734375 0.2900390609375 0.8559374999999999 0.1943359390625 0.7051171859375 0.07421874687499999 0.6045703140625001 0.0546875 0.380273440625 0.0605468734375 0.290039059375 0.08984374687499999 0.1263281265625 0.2490234390625 0.08249999999999999 0.4189453125 0.0953906265625 0.6025390609375 0.1521093734375 0.6611328125 0.2139843734375 0.8076171875 0.266835940625 0.8574218734375 0.41636718593750005 0.890625 0.6226171859375 0.8691406265625 0.6329296859375 0.890625 0.675468746875 0.8955078125 0.7837500000000001 0.8017578125 0.8946093734374999 0.5849609390625 0.9203906265624999 0.4443359390625 0.9049218734375 0.2900390609375
4 0.6007031265625 0.4814453125 0.6367968734375 0.4208984390625 0.654843746875 0.3662109390625 0.6522656265625 0.3349609390625 0.6329296859375 0.316406253125 0.6123046859375 0.3183593734375 0.53625 0.3857421875 0.5130468734375 0.4912109390625 0.520781253125 0.6376953125 0.578789059375 0.6640625 0.603281253125 0.6630859390625 0.6110156265625 0.6376953125 0.5775 0.5986328125 0.5749218734375 0.5634765609375 0.6007031265625 0.4814453125
4 0.489843746875 0.3896484390625 0.413789059375 0.3261718734375 0.3751171859375 0.316406253125 0.35578125312500003 0.3408203125 0.3532031265625 0.3798828125 0.4125 0.4970703125 0.43054687343750003 0.5654296875 0.427968746875 0.6005859390625 0.3944531265625 0.6396484390625 0.397031253125 0.6708984390625 0.4150781265625 0.6728515609375 0.48082031406250003 0.6484375 0.4975781265625 0.6279296875 0.5053124999999999 0.4541015609375 0.489843746875 0.3896484390625


================================================
FILE: TumorDetection/train/labels/no_tumor_1264_jpg.rf.3a4558975d3e7aa529cc9e050b1e1848.txt
================================================
0 0.7089843734375 0.52141461875 0.6855468734375 0.34702845781249997 0.6347656265625 0.1935686375 0.5810546859375 0.11509486875 0.4755859359375 0.09765625 0.3994140640625 0.1499720953125 0.32421875 0.3400530125 0.2988281265625 0.497000559375 0.2949218734375 0.65046038125 0.32421875 0.7864815828125 0.3740234359375 0.87890625 0.4345703140625 0.9416852671875 0.50390625 0.9608677437500001 0.5986328140625 0.9207589281249999 0.6464843734375 0.8492606015625 0.703125 0.6888253328125 0.7089843734375 0.52141461875


================================================
FILE: TumorDetection/train/labels/no_tumor_1265_jpg.rf.7c655129f792360998c6cfc59a897dbd.txt
================================================
0 0.6640625 0.8115234375 0.748046875 0.6845703125 0.763671875 0.5087890625 0.720703125 0.2978515625 0.669921875 0.1982421875 0.5791015625 0.150390625 0.4404296875 0.140625 0.390625 0.1689453125 0.4140625 0.1748046875 0.390625 0.1806640625 0.4072265625 0.1953125 0.3779296875 0.1953125 0.3759765625 0.181640625 0.345703125 0.2001953125 0.3564453125 0.232421875 0.2998046875 0.232421875 0.2958984375 0.28125 0.28125 0.2626953125 0.265625 0.3974609375 0.2529296875 0.3828125 0.236328125 0.4150390625 0.224609375 0.5205078125 0.251953125 0.6611328125 0.2744140625 0.640625 0.298828125 0.6591796875 0.283203125 0.6865234375 0.29296875 0.7119140625 0.27734375 0.7158203125 0.30078125 0.7783203125 0.3955078125 0.857421875 0.482421875 0.8681640625 0.5009765625 0.845703125 0.5244140625 0.865234375 0.5830078125 0.86328125 0.6640625 0.8115234375


================================================
FILE: TumorDetection/train/labels/no_tumor_126_jpg.rf.50e10f563b8a940f1503c7dfb8cb6021.txt
================================================
0 0.8378906265625 0.5126953140625 0.8261718734375 0.4033203140625 0.765625 0.2294921859375 0.6728515640625 0.13671875 0.5244140640625 0.08007812656249999 0.3857421859375 0.09765625 0.26171875 0.1806640640625 0.1972656265625 0.2861328140625 0.1464843734375 0.4794921859375 0.1738281265625 0.6923828140625 0.2324218734375 0.8330078140625 0.3544921859375 0.92578125 0.4824218734375 0.9462890640625 0.6083984359375 0.91796875 0.7285156265625 0.8154296859375 0.80078125 0.6787109359375 0.8378906265625 0.5126953140625


================================================
FILE: TumorDetection/train/labels/no_tumor_1271_jpg.rf.99b32784706552b3695ed804e917a16e.txt
================================================
0 0.8545421484375 0.6279296875 0.8545421484375 0.4306640640625 0.8056504375 0.2763671875 0.7195585015625 0.1582031234375 0.645158065625 0.10546875156249999 0.5728833578125 0.08203124843750001 0.3836936796875 0.08984375156249999 0.23489280468749998 0.171875 0.14667514375 0.3037109359375 0.0914062515625 0.4912109359375 0.0935319765625 0.6572265640625 0.1551780515625 0.7861328125 0.2816587953125 0.8867187515625 0.45490552500000003 0.9248046875 0.596266353125 0.9101562484375 0.7046784140625 0.8574218765625 0.8035247062499999 0.7431640640625 0.8545421484375 0.6279296875


================================================
FILE: TumorDetection/train/labels/no_tumor_1278_jpg.rf.9034ecaec68ef1ff513b5ff8dcee2f29.txt
================================================
0 0.7910156265625 0.5009765640625 0.7597656265625 0.3955078140625 0.7539062484375 0.2880859375 0.6816406265625 0.1806640625 0.5654296890625 0.123046875 0.5263671859375 0.125 0.5029296890625 0.1484375015625 0.4365234359375 0.11523437343750001 0.3388671859375 0.1484375015625 0.251953125 0.2841796890625 0.2109375015625 0.4501953109375 0.2089843734375 0.5927734359375 0.2578124984375 0.7431640625 0.3066406265625 0.8173828140625 0.4228515640625 0.8964843734375 0.5273437515625 0.9150390625 0.6259765640625 0.8789062484375 0.71875 0.7861328140625 0.7871093734375 0.6416015640625 0.7910156265625 0.5009765640625


================================================
FILE: TumorDetection/train/labels/no_tumor_1288_jpg.rf.1716523f611f167a31ab8005753b985a.txt
================================================
0 0.846366453125 0.4912109375 0.7783940140625 0.3701171890625 0.7257701937500001 0.3330078109375 0.7476967875 0.2919921890625 0.7235775375 0.2158203109375 0.6720500421875 0.1738281265625 0.6281968578125 0.1757812515625 0.5996922921875 0.140625 0.4790960328125 0.1289062515625 0.37823371250000004 0.1503906265625 0.35521079218749996 0.1923828109375 0.2894310140625 0.2314453109375 0.271889740625 0.2724609375 0.2982016515625 0.3134765625 0.2773713875 0.3085937484375 0.23242187812499998 0.3427734375 0.1666420984375 0.5048828109375 0.1666420984375 0.5791015625 0.2302292171875 0.7236328109375 0.2861420234375 0.7734375 0.4067382828125 0.8339843734375 0.4505914671875 0.796875 0.46155475937499996 0.8183593734375 0.53500884375 0.8388671890625 0.650123453125 0.8027343734375 0.7718160375 0.7158203109375 0.831017834375 0.6142578109375 0.846366453125 0.4912109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1293_jpg.rf.3df5863ade1128b66f3e6b4f2b8f315e.txt
================================================
0 0.8398437515625 0.6669921859375 0.8496093734375 0.3994140625 0.7734375015625 0.2275390625 0.6884765640625 0.138671875 0.5673828140625 0.08593750156249999 0.4267578140625 0.08203124843750001 0.2763671859375 0.1523437515625 0.1816406265625 0.2724609375 0.1523437515625 0.3505859375 0.1484375015625 0.5693359375 0.1835937515625 0.7431640625 0.2587890625 0.8398437515625 0.4052734359375 0.9140624984375 0.5410156265625 0.9169921859375 0.6689453109375 0.8867187515625 0.7548828140625 0.8242187515625 0.798828125 0.7666015640625 0.8398437515625 0.6669921859375


================================================
FILE: TumorDetection/train/labels/no_tumor_1297_jpg.rf.cebfcd4925e57647d14e1a8e00483656.txt
================================================
0 0.9692610984375 0.4365234359375 0.9024897749999999 0.2294921859375 0.8044867109375 0.11328125 0.6450971078125 0.03515625 0.4232440140625 0.02734375 0.304778765625 0.0625 0.2358535328125 0.10351562656249999 0.13138872968749998 0.2041015640625 0.051693924999999995 0.3876953140625 0.053847840625 0.5673828140625 0.0990800234375 0.7255859359375 0.1658513453125 0.8193359359375 0.22939179375 0.875 0.3586266078125 0.9511718734375 0.542786215625 0.9599609359375 0.707560603125 0.9140625 0.842180196875 0.7919921859375 0.9520297890625 0.5986328140625 0.9692610984375 0.4365234359375


================================================
FILE: TumorDetection/train/labels/no_tumor_1309_jpg.rf.e92527e498844db62f674a25860351b4.txt
================================================
0 0.6201171875 0.884765625 0.6953125 0.8154296875 0.759765625 0.7158203125 0.7734375 0.6826171875 0.76953125 0.6474609375 0.7890625 0.6220703125 0.798828125 0.5751953125 0.787109375 0.4697265625 0.76171875 0.4267578125 0.7578125 0.3994140625 0.734375 0.3779296875 0.734375 0.3037109375 0.71875 0.2587890625 0.697265625 0.2392578125 0.68359375 0.2041015625 0.6650390625 0.1875 0.5458984375 0.142578125 0.5263671875 0.14453125 0.5009765625 0.166015625 0.4755859375 0.150390625 0.3994140625 0.15625 0.3701171875 0.16796875 0.3203125 0.2041015625 0.279296875 0.2626953125 0.2734375 0.3603515625 0.2421875 0.4189453125 0.240234375 0.4501953125 0.2109375 0.5224609375 0.2109375 0.6728515625 0.265625 0.8037109375 0.3466796875 0.859375 0.3701171875 0.859375 0.3857421875 0.884765625 0.484375 0.8994140625 0.4990234375 0.890625 0.5439453125 0.8984375 0.5986328125 0.89453125 0.6201171875 0.884765625


================================================
FILE: TumorDetection/train/labels/no_tumor_1325_jpg.rf.8874659d760783bb23c854f7112a65c6.txt
================================================
0 0.8443444281250001 0.6396484390625 0.8489583328125001 0.5576171875 0.8881765187499999 0.4443359390625 0.885869565625 0.3779296875 0.8581861406250001 0.2919921875 0.80281929375 0.2294921875 0.695546025 0.150390625 0.5409802421875 0.119140625 0.49253425 0.1445312515625 0.453316065625 0.115234375 0.3794936015625 0.111328125 0.307978090625 0.1328125 0.21800696406250003 0.1796875 0.11996150312500001 0.2802734390625 0.083050271875 0.3779296875 0.0945850328125 0.5439453125 0.14764492812500002 0.7333984390625 0.1937839671875 0.7998046875 0.25030429062500004 0.8515625 0.291829428125 0.880859375 0.38180055468749996 0.9101562515625 0.446395209375 0.9140625 0.4763855875 0.904296875 0.55597543125 0.9150390609375 0.6632487 0.880859375 0.7289968296875 0.8310546875 0.7659080609375 0.7841796875 0.8443444281250001 0.6396484390625


================================================
FILE: TumorDetection/train/labels/no_tumor_1326_jpg.rf.c8d7b79cda542c39fd6eb18013033052.txt
================================================
0 0.787109375 0.3271484375 0.6904296875 0.1484375 0.5498046875 0.0859375 0.4384765625 0.0859375 0.3095703125 0.138671875 0.23828125 0.2275390625 0.193359375 0.3974609375 0.1953125 0.5537109375 0.24609375 0.7470703125 0.29296875 0.8251953125 0.4013671875 0.896484375 0.546875 0.9052734375 0.6591796875 0.857421875 0.7421875 0.7412109375 0.79296875 0.5400390625 0.787109375 0.3271484375


================================================
FILE: TumorDetection/train/labels/no_tumor_1330_jpg.rf.abd26941bb64c1899fa212b1f68d3f7f.txt
================================================
0 0.87890625 0.41408017656249996 0.8222656234375 0.2387761578125 0.7275390609375 0.11686934375 0.5927734390625 0.0423147609375 0.4638671890625 0.040299773437500006 0.3701171890625 0.0785845609375 0.2783203109375 0.153139140625 0.23046875 0.2206412625 0.18359375 0.3919152984375 0.18359375 0.5510994078125 0.23046875 0.7102835109375001 0.30859375 0.8573776875 0.4189453109375 0.9309247734375001 0.546875 0.939992221875 0.6474609390625 0.9188348421875 0.75390625 0.8251378687500001 0.8730468765625 0.5390094750000001 0.87890625 0.41408017656249996


================================================
FILE: TumorDetection/train/labels/no_tumor_1346_jpg.rf.2e4aff010471355370cfa24a13c46516.txt
================================================
0 0.7451171859375 0.7617370890624999 0.7578125 0.7671471546874999 0.7480468734375 0.7173745593749999 0.7626953140625 0.6795041078125 0.78515625 0.67842209375 0.77734375 0.6611098875 0.8046875 0.57887690625 0.8066406265625 0.3732944546875 0.75390625 0.234796803125 0.7041015640625 0.1687940140625 0.6591796859375 0.12984155 0.6533203140625 0.14715375625 0.6279296859375 0.11685739375 0.5888671859375 0.11685739375 0.5849609359375 0.0973811640625 0.5576171859375 0.09305310781250001 0.5332031265625 0.09629915 0.5263671859375 0.14715375625 0.51953125 0.0984631734375 0.5009765640625 0.090889084375 0.4072265640625 0.11685739375 0.3046875 0.23263277499999999 0.2578125 0.3538182203125 0.2548828140625 0.48257775625 0.23046875 0.481495746875 0.21484375 0.425231075 0.2109375 0.598353140625 0.2285156265625 0.6546178125 0.2607421859375 0.6730120281249999 0.265625 0.62432145 0.2480468734375 0.6091732671875 0.2714843734375 0.5961891125000001 0.2753906265625 0.669765990625 0.3857421859375 0.859118253125 0.4462890640625 0.9110548734374999 0.5146484359375 0.9153829203124999 0.5185546859375 0.947843309375 0.546875 0.9489253234374999 0.6533203140625 0.913218896875 0.69921875 0.819083775 0.7421875 0.786623384375 0.7451171859375 0.7617370890624999


================================================
FILE: TumorDetection/train/labels/no_tumor_1351_jpg.rf.315a5f4d14e57f0c2ac119c53b2f93b0.txt
================================================
0 0.6777343734375 0.8212890625 0.75 0.7119140625 0.7871093734375 0.4951171859375 0.7851562484375 0.3525390625 0.7109375015625 0.1962890625 0.6240234359375 0.11914062656249999 0.5244140625 0.08984375156249999 0.4228515640625 0.09960937343750001 0.3046875015625 0.1806640625 0.232421875 0.3095703109375 0.2070312484375 0.4111328140625 0.21875 0.5908203109375 0.2890624984375 0.7861328140625 0.3388671859375 0.8496093734375 0.453125 0.8837890625 0.5009765640625 0.8535156265625 0.5302734359375 0.875 0.5654296890625 0.875 0.6777343734375 0.8212890625


================================================
FILE: TumorDetection/train/labels/no_tumor_1356_jpg.rf.84735d5194d7ebaa17bc10f797435335.txt
================================================
0 0.8476562484375 0.4287109375 0.8027343734375 0.2587890625 0.7119140625 0.1445312484375 0.6005859375 0.08398437343750001 0.4462890625 0.07421875156249999 0.3408203109375 0.10546875156249999 0.2246093734375 0.2119140625 0.1796875015625 0.3037109375 0.154296875 0.4912109375 0.1777343734375 0.6044921859375 0.2753906265625 0.8232421859375 0.3642578140625 0.8945312484375 0.5273437515625 0.9189453109375 0.6298828140625 0.8984375015625 0.7285156265625 0.8212890625 0.8320312484375 0.5869140625 0.8476562484375 0.4287109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1362_jpg.rf.229adcfa099b098f15ea935f83547cbc.txt
================================================
0 0.70703125 0.660923546875 0.70703125 0.497000559375 0.6777343734375 0.319126671875 0.6279296859375 0.1813616078125 0.5966796859375 0.1325334828125 0.5439453140625 0.1011439765625 0.4541015640625 0.104631696875 0.4033203140625 0.146484375 0.3261718734375 0.3400530125 0.2988281265625 0.510951453125 0.2988281265625 0.6818498875000001 0.33984375 0.8213588156249999 0.4189453140625 0.927734375 0.5 0.9608677437500001 0.5498046859375 0.9486607140625001 0.6201171859375 0.8928571421875 0.6699218734375 0.789969309375 0.70703125 0.660923546875


================================================
FILE: TumorDetection/train/labels/no_tumor_1375_jpg.rf.1cc6e4c7fced24a8e2b20c22761f5e10.txt
================================================
0 0.9755907953125 0.7626953109375 0.9926636375 0.6884765625 0.9926636375 0.4794921875 0.9609569359375 0.3525390640625 0.8780317171875 0.2021484375 0.7573023531249999 0.12109374843750001 0.586573965625 0.068359375 0.40121171718750004 0.07421874843750001 0.2182884421875 0.150390625 0.148777596875 0.2119140640625 0.10975396562499999 0.2783203109375 0.0512185171875 0.4306640640625 0.024389771875 0.6201171875 0.0609744234375 0.7900390640625 0.1414606671875 0.9033203109375 0.29877468125 0.9863281234375 0.34755421875 1 0.6926694624999999 0.9990234375 0.7573023531249999 0.984375 0.8390080843750001 0.9384765625 0.9365671640625 0.8427734375 0.9755907953125 0.7626953109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1376_jpg.rf.aff2f8a1edbd73173f348c43503fb461.txt
================================================
0 0.9203906265624999 0.5576171875 0.943593746875 0.5537109390625 0.9178124999999999 0.5244140609375 0.953906253125 0.5224609390625 0.9178124999999999 0.5107421875 0.9526171859374999 0.5078125 0.9564843734374999 0.4892578125 0.9178124999999999 0.4775390609375 0.953906253125 0.4716796875 0.916523440625 0.472656253125 0.9100781265625001 0.4384765609375 0.953906253125 0.4365234390625 0.9564843734374999 0.3818359390625 0.9075 0.3623046875 0.926835940625 0.3515625 0.8559374999999999 0.3330078125 0.8868750000000001 0.3251953125 0.8378906265625 0.3037109390625 0.830156253125 0.2587890609375 0.8095312531250001 0.2548828125 0.8327343734375001 0.2451171875 0.7283203140625 0.1640625 0.599414059375 0.125 0.5272265593750001 0.140625 0.4189453140625 0.125 0.349335940625 0.1777343734375 0.2951953140625 0.1640625 0.242343746875 0.2001953125 0.2578125 0.2119140609375 0.18562499999999998 0.2587890609375 0.1366406265625 0.3818359390625 0.0515625 0.4462890609375 0.055429685937499994 0.4707031265625 0.073476559375 0.4453125 0.087656253125 0.4541015609375 0.08249999999999999 0.4794921875 0.046406253125000005 0.4853515609375 0.0476953140625 0.53125 0.061875 0.5009765609375 0.0683203140625 0.515625 0.0850781265625 0.5009765609375 0.06445312656249999 0.5302734390625 0.061875 0.6962890609375 0.144375 0.8095703125 0.279726559375 0.9160156265625 0.372539059375 0.9433593734375 0.5671875 0.9462890609375 0.751523440625 0.886718746875 0.8533593734375 0.8017578125 0.9332812531249999 0.5849609390625 0.9203906265624999 0.5576171875
4 0.4705078140625 0.355468746875 0.43957031406249997 0.3457031265625 0.4253906265625 0.3525390609375 0.4202343734375 0.3701171875 0.433125 0.3779296875 0.42281250000000004 0.3837890609375 0.433125 0.3935546875 0.42281250000000004 0.4033203125 0.43054687343750003 0.4091796875 0.4253906265625 0.4365234390625 0.433125 0.4443359390625 0.4253906265625 0.4501953125 0.43570312656250004 0.4560546875 0.43570312656250004 0.4736328125 0.4266796859375 0.4824218734375 0.417656253125 0.4716796875 0.4021875 0.5205078125 0.35578125312500003 0.5986328125 0.345468746875 0.6396484390625 0.3532031265625 0.6806640609375 0.3892968734375 0.6845703125 0.47437500000000005 0.6064453125 0.48210937343750004 0.5810546875 0.48210937343750004 0.3720703125 0.4705078140625 0.355468746875
4 0.5749218734375 0.4423828125 0.5749218734375 0.3955078125 0.6058593734375 0.3583984390625 0.5916796859375 0.3222656265625 0.5401171859375 0.3378906265625 0.4975781265625 0.3642578125 0.4924218734375 0.6044921875 0.5336718734375 0.6298828125 0.537539059375 0.6464843734375 0.556875 0.6513671875 0.565898440625 0.6738281265625 0.596835940625 0.6855468734375 0.6316406265625 0.6865234390625 0.6522656265625 0.6552734390625 0.6471093734375 0.6123046875 0.5955468734375 0.5205078125 0.5749218734375 0.4423828125


================================================
FILE: TumorDetection/train/labels/no_tumor_1377_jpg.rf.f1b71258d44b9c64d7d2ead7ca7db5d2.txt
================================================
0 0.859375 0.3408203109375 0.8007812484375 0.1357421859375 0.7451171859375 0.07421875156249999 0.6669921859375 0.041015626562500004 0.3857421859375 0.03125 0.2900390625 0.076171875 0.2285156265625 0.1591796890625 0.189453125 0.2744140625 0.140625 0.5439453109375 0.138671875 0.6943359375 0.1992187515625 0.8310546890625 0.3369140625 0.951171875 0.4248046890625 0.986328125 0.5546875015625 0.9892578140625 0.7080078140625 0.9414062484375 0.8339843734375 0.8232421859375 0.875 0.6416015640625 0.859375 0.3408203109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1378_jpg.rf.c502c2a66eead244e8ca6d8131d565cd.txt
================================================
0 0.995304990625 0.7666015625 0.995304990625 0.3037109375 0.95440204375 0.2041015625 0.87032376875 0.08300781406249999 0.7601130578124999 0 0.255643403125 0.0039062484375 0.163611778125 0.06738281406249999 0.07498873125 0.1708984375 0.0068171593749999995 0.3251953140625 0.00908954375 0.8154296859375 0.0818058890625 0.9130859375 0.18747183125 1 0.7998798078125 0.9990234375 0.8907752406249999 0.9345703140625 0.939631534375 0.8652343765625 0.9453125 0.8798828140625 0.995304990625 0.7666015625


================================================
FILE: TumorDetection/train/labels/no_tumor_1381_jpg.rf.78b44b23406690cab08efe18368e6604.txt
================================================
0 0.9925781265625 0.7138671875 0.9874218734375001 0.4150390609375 0.8765625 0.1923828125 0.7901953140625 0.105468746875 0.6148828140625 0.021484373437500003 0.4060546859375 0.011718746875 0.22300781406249998 0.08203125312500001 0.07476562656249999 0.2451171875 0.010312499999999999 0.4111328125 0.020624999999999998 0.6865234390625 0.056718746875 0.8193359390625 0.14050781406249999 0.894531253125 0.287460940625 0.9667968734375 0.4292578140625 0.996093746875 0.5336718734375 0.9951171875 0.6664453140625 0.9863281265625 0.805664059375 0.9375 0.9461718734375 0.8251953125 0.9925781265625 0.7138671875


================================================
FILE: TumorDetection/train/labels/no_tumor_1382_jpg.rf.e8ca4a15c262a636d8b9c58b2277075a.txt
================================================
0 0.8164062484375 0.4638671859375 0.7753906265625 0.3232421859375 0.6923828140625 0.1933593734375 0.5830078140625 0.1289062484375 0.4443359375 0.11914062656249999 0.3212890625 0.1679687515625 0.2089843734375 0.3056640625 0.1660156265625 0.4521484359375 0.169921875 0.7275390625 0.2148437515625 0.8291015640625 0.3115234359375 0.9277343734375 0.4033203109375 0.9746093734375 0.4941406265625 0.9833984359375 0.6318359375 0.951171875 0.7753906265625 0.8193359375 0.8066406265625 0.7353515640625 0.8164062484375 0.4638671859375


================================================
FILE: TumorDetection/train/labels/no_tumor_1383_jpg.rf.83d5b9dea50a4b1c805b3dfeb523b25c.txt
================================================
0 0.7802734390625 0.6609162890625 0.78515625 0.674013715625 0.7949218765625 0.5107996328125 0.7441406234375 0.24683611249999998 0.69921875 0.1601916015625 0.5869140609375 0.086644515625 0.4658203109375 0.06850961562499999 0.3369140609375 0.10074943437499999 0.2753906234375 0.15011665624999998 0.2363281234375 0.2125813078125 0.1640625 0.41005019843749996 0.1484375 0.6921486156250001 0.17578125 0.7122985015625 0.16015625 0.72035845625 0.1796875 0.7284184125000001 0.1640625 0.74050834375 0.1875 0.744538321875 0.17578125 0.7525982765625 0.1933593765625 0.7646882078125 0.1796875 0.7727481625 0.20703125 0.790883059375 0.1972656234375 0.8029729906249999 0.2167968765625 0.8110329453125 0.2128906234375 0.8271528546875 0.2744140609375 0.886595021875 0.4189453109375 0.9611496015625001 0.49609375 0.9641720859375 0.6318359390625 0.9289097828124999 0.7109375 0.8513327187499999 0.74609375 0.788868071875 0.7802734390625 0.6609162890625


================================================
FILE: TumorDetection/train/labels/no_tumor_1401_jpg.rf.000dba443c62539495f338739308ce48.txt
================================================
0 0.841796875 0.5927734359375 0.841796875 0.4404296890625 0.8222656265625 0.4248046890625 0.8046875015625 0.2919921859375 0.7285156265625 0.1533203109375 0.6376953109375 0.078125 0.3798828140625 0.080078125 0.2724609375 0.1484375015625 0.2226562484375 0.2060546890625 0.1972656265625 0.3115234359375 0.1640624984375 0.5908203109375 0.21875 0.8173828140625 0.3134765640625 0.9121093734375 0.4609375015625 0.9580078140625 0.6005859375 0.9453124984375 0.7294921859375 0.875 0.796875 0.7724609375 0.841796875 0.5927734359375


================================================
FILE: TumorDetection/train/labels/no_tumor_1402_jpg.rf.a1a01c9623bf180dc9b357d86dc4291f.txt
================================================
0 0.8496093734375 0.4072265640625 0.7871093734375 0.2529296890625 0.6865234359375 0.1367187515625 0.5673828140625 0.08593750156249999 0.4130859375 0.08593750156249999 0.2822265640625 0.1484375015625 0.1835937515625 0.2685546890625 0.1523437515625 0.3525390625 0.1484375015625 0.5615234359375 0.1835937515625 0.7392578140625 0.2451171859375 0.8242187515625 0.4013671859375 0.908203125 0.533203125 0.9130859375 0.6572265640625 0.8867187515625 0.779296875 0.7900390625 0.8378906265625 0.6533203109375 0.8496093734375 0.4072265640625


================================================
FILE: TumorDetection/train/labels/no_tumor_1417_jpg.rf.dc51cc6b0a36c02be55b38e6b9b32867.txt
================================================
0 0.8378688640625 0.8388671875 0.8745103593750001 0.7880859375 0.92092291875 0.6923828125 0.942907815625 0.6279296875 0.9477933484375001 0.5576171875 0.942907815625 0.4462890625 0.9038235546875001 0.3232421875 0.810998434375 0.1572265640625 0.7511506578125 0.10156249843750001 0.704738096875 0.07421875 0.633897871875 0.048828125 0.54595828125 0.033203125 0.41404889999999994 0.037109373437499996 0.34809420625 0.05078125 0.24549801875 0.08984375 0.1807647109375 0.1357421875 0.131909384375 0.2099609375 0.063511925 0.3505859375 0.04396979375 0.4248046875 0.0366414953125 0.5205078125 0.0537408609375 0.6787109375 0.13923768281249999 0.8408203125 0.22595588906250003 0.908203125 0.326109309375 0.94921875 0.4262627296875 0.966796875 0.5203092375 0.9677734359375 0.5825997765625 0.9628906265625 0.6607683015625 0.94140625 0.7731355546875001 0.8847656265625 0.8378688640625 0.8388671875


================================================
FILE: TumorDetection/train/labels/no_tumor_1422_jpg.rf.f3bccc45e3099fbc62a02c4e79bc41e8.txt
================================================
0 0.6240305140625 0.9101562515625 0.7093877390625 0.8828125 0.7543732999999999 0.8525390609375 0.8489583328125001 0.7412109390625 0.9066321343749999 0.5361328125 0.9158599421875 0.3720703125 0.881255659375 0.2822265609375 0.7809032499999999 0.177734375 0.6217235609375 0.111328125 0.5479010984375 0.11328125156249999 0.5040690109375 0.1484375 0.47177168125 0.12109374843750001 0.3910283625 0.125 0.30567113749999997 0.1484375 0.1937839671875 0.2314453125 0.143031021875 0.2880859390625 0.11304064843749999 0.3837890609375 0.11073369531249999 0.4365234390625 0.145337975 0.5400390609375 0.1591796890625 0.6474609390625 0.2353091015625 0.7861328125 0.28606204687500003 0.8486328125 0.3425823703125 0.884765625 0.4590834453125 0.9169921875 0.5063759640625001 0.9140625 0.5271385296875 0.900390625 0.559435859375 0.916015625 0.6240305140625 0.9101562515625


================================================
FILE: TumorDetection/train/labels/no_tumor_1430_jpg.rf.f747c66cb2015e28af93695f56c4123c.txt
================================================
0 0.501953125 0.9117739359374999 0.5263671875 0.9204574953125 0.5498046875 0.9030903734375 0.5810546875 0.9030903734375 0.591796875 0.8805131140625001 0.5927734375 0.830148459375 0.6064453125 0.84056873125 0.625 0.831885171875 0.625 0.7971509250000001 0.654296875 0.7311558609375 0.666015625 0.6860013406250001 0.658203125 0.644320246875 0.66796875 0.6130594265625 0.65234375 0.58179860625 0.6689453125 0.5765884703124999 0.6796875 0.5505377859375 0.677734375 0.5019098421875 0.69140625 0.44633504999999996 0.69140625 0.42202107812499995 0.68359375 0.40118053124999997 0.689453125 0.3386588890625 0.662109375 0.22403588124999999 0.6298828125 0.156304103125 0.6044921875 0.1250432828125 0.5830078125 0.121569859375 0.5419921875 0.0903090359375 0.5087890625 0.0903090359375 0.4873046875 0.121569859375 0.4697265625 0.08336218906250001 0.4228515625 0.0868356125 0.3876953125 0.10767616093750002 0.326171875 0.2031953359375 0.32421875 0.23098273125 0.330078125 0.241403003125 0.30859375 0.258770128125 0.302734375 0.2796106734375 0.306640625 0.2969777953125 0.294921875 0.3212917671875 0.29296875 0.3733931359375 0.30859375 0.39423368124999997 0.291015625 0.42896792812499995 0.291015625 0.4602287484375 0.296875 0.512330115625 0.30859375 0.5331706625 0.3046875 0.547064359375 0.30859375 0.5852720312499999 0.32421875 0.6026391531249999 0.322265625 0.6304265484375 0.333984375 0.644320246875 0.32421875 0.6651607953124999 0.328125 0.7068418890625 0.341796875 0.75546983125 0.37109375 0.8006243515625 0.3671875 0.8249383218749999 0.373046875 0.8457788703125001 0.3818359375 0.8579358562499999 0.4111328125 0.8509890062500001 0.4140625 0.8944068109375 0.4384765625 0.9204574953125 0.490234375 0.9291410578124999 0.4990234375 0.8370953078125 0.5078125 0.8423054453125 0.5 0.8874599640625 0.501953125 0.9117739359374999


================================================
FILE: TumorDetection/train/labels/no_tumor_1432_jpg.rf.7adef0d175984efb42ffcd714fd528b8.txt
================================================
0 0.75390625 0.5288085921875 0.7558593765625 0.3295898421875 0.7480468765625 0.2768554671875 0.7089843765625 0.1889648421875 0.6318359375 0.07324218593750001 0.5732421875 0.041015625 0.4384765625 0.03515625 0.3564453125 0.08203125 0.2714843765625 0.2006835921875 0.25 0.2592773421875 0.234375 0.3676757828125 0.2421875 0.5786132828125 0.26953125 0.7338867171875 0.3564453125 0.8876953140625 0.3876953125 0.9228515640625 0.4423828125 0.955078125 0.5253906234375 0.9624023421875 0.5849609375 0.94921875 0.6279296875 0.9228515640625 0.72265625 0.7602539078125 0.75390625 0.5288085921875


================================================
FILE: TumorDetection/train/labels/no_tumor_1435_jpg.rf.4d1db0e9166cc45301b63de06e43ab13.txt
================================================
0 0.8902196328124999 0.6083984375 0.90776090625 0.3427734375 0.75537109375 0.1835937484375 0.6588940890625 0.1308593734375 0.5821510187500001 0.11328125156249999 0.413316259375 0.11523437343750001 0.2861420234375 0.1601562515625 0.184183371875 0.2412109375 0.083321046875 0.3759765625 0.12059625468749999 0.6650390625 0.16006412187500002 0.7587890625 0.2214585796875 0.8330078109375 0.2773713875 0.8710937484375 0.3760410515625 0.9121093734375 0.46484375 0.9169921890625 0.619426221875 0.9082031265625 0.744407796875 0.8671875 0.8441737921874999 0.7509765625 0.8902196328124999 0.6083984375


================================================
FILE: TumorDetection/train/labels/no_tumor_1447_jpg.rf.196ca26893868d7b806177223b3913a9.txt
================================================
0 0.7539062515625 0.8017578125 0.791015625 0.7138671875 0.8125 0.6318359375 0.833984375 0.4951171875 0.833984375 0.3857421875 0.814453125 0.2783203125 0.7929687484375 0.2119140625 0.7695312515625 0.1708984375 0.7236328125 0.1171875 0.6318359375 0.06640625156249999 0.5849609375 0.0507812515625 0.4970703125 0.041015625 0.4169921875 0.044921875 0.3740234375 0.0585937484375 0.2861328125 0.1015625 0.248046875 0.1396484375 0.216796875 0.1845703125 0.1914062515625 0.2548828125 0.1679687484375 0.3544921875 0.1640625 0.4384765625 0.169921875 0.5068359375 0.2148437484375 0.7138671875 0.248046875 0.7919921875 0.2773437484375 0.8310546875 0.3427734375 0.8867187484375 0.3896484375 0.912109375 0.4521484375 0.9257812515625 0.5625 0.9267578125 0.6123046875 0.9179687484375 0.6708984375 0.888671875 0.724609375 0.8447265625 0.7539062515625 0.8017578125


================================================
FILE: TumorDetection/train/labels/no_tumor_1448_jpg.rf.ced72720cad2eba824730db111df41c7.txt
================================================
0 0.5888671875 0.8509890062500001 0.6201171875 0.8544624328125 0.630859375 0.831885171875 0.630859375 0.8006243515625 0.658203125 0.75546983125 0.67578125 0.6894747671875 0.67578125 0.6651607953124999 0.66796875 0.644320246875 0.6796875 0.626953125 0.67578125 0.6026391531249999 0.6865234375 0.5974290171875001 0.69140625 0.58179860625 0.69140625 0.5331706625 0.703125 0.512330115625 0.708984375 0.4602287484375 0.708984375 0.4324413515625 0.69140625 0.39423368124999997 0.70703125 0.3733931359375 0.705078125 0.3212917671875 0.6953125 0.293504371875 0.6953125 0.26919039843750003 0.673828125 0.241403003125 0.673828125 0.2031953359375 0.6103515625 0.10767616093750002 0.5830078125 0.0903090359375 0.5263671875 0.0868356125 0.5126953125 0.1250432828125 0.4873046875 0.0903090359375 0.4482421875 0.0937824625 0.3974609375 0.121569859375 0.3662109375 0.1597775296875 0.337890625 0.22403588124999999 0.310546875 0.32823861718749997 0.31640625 0.40118053124999997 0.30859375 0.42202107812499995 0.30859375 0.4498084734375 0.3203125 0.49843641875 0.322265625 0.5609580578125 0.3330078125 0.5765884703124999 0.349609375 0.57485175625 0.33203125 0.61653285 0.34375 0.6477936734374999 0.33203125 0.67558106875 0.345703125 0.7311558609375 0.375 0.7971509250000001 0.375 0.831885171875 0.3935546875 0.84056873125 0.4072265625 0.830148459375 0.408203125 0.8805131140625001 0.4228515625 0.9065637984375 0.4501953125 0.9030903734375 0.4873046875 0.9239309203125 0.4990234375 0.9030903734375 0.5107421875 0.9308777703125 0.529296875 0.9291410578124999 0.5576171875 0.9239309203125 0.5810546875 0.9030903734375 0.58984375 0.8805131140625001 0.5888671875 0.8509890062500001


================================================
FILE: TumorDetection/train/labels/no_tumor_1454_jpg.rf.916adb111444c20987e20200b9798eb7.txt
================================================
0 0.6755539781250001 0.791015625 0.7262500000000001 0.7568359375 0.7663352265625 0.6943359375 0.7757670453125 0.6513671875 0.799346590625 0.6162109375 0.8182102265625 0.5693359375 0.84650568125 0.4755859375 0.8535795453125001 0.4013671875 0.8252840906250001 0.3037109375 0.778125 0.2294921875 0.7439346578125 0.197265625 0.6873437484375 0.162109375 0.614247159375 0.130859375 0.562372159375 0.1328125 0.5411505671875 0.142578125 0.539971590625 0.1533203125 0.5317187484375 0.154296875 0.4987073859375 0.130859375 0.4609801125 0.126953125 0.4020312515625 0.1484375 0.35251420312500004 0.177734375 0.2923863640625 0.2294921875 0.25465909062500003 0.2900390625 0.21457386406249998 0.4091796875 0.212215909375 0.4541015625 0.22400568125000003 0.5517578125 0.2617329546875 0.6396484375 0.2735227265625 0.7099609375 0.30181818125 0.7646484375 0.3136079546875 0.7724609375 0.30889204531250003 0.7802734375 0.332471590625 0.7861328125 0.3242187484375 0.78515625 0.3183238640625 0.7919921875 0.33954545468749997 0.8017578125 0.33365056718749997 0.810546875 0.3407244328125 0.806640625 0.361946021875 0.8203125 0.36666193125 0.810546875 0.373735796875 0.814453125 0.4987073859375 0.810546875 0.4928125 0.8037109375 0.5104971593750001 0.798828125 0.5128551125 0.810546875 0.56119318125 0.8212890625 0.5835937484375 0.814453125 0.6566903421875 0.810546875 0.6496164765625 0.80078125 0.6755539781250001 0.791015625


================================================
FILE: TumorDetection/train/labels/no_tumor_1457_jpg.rf.d5ebbc52838c88699a3a448e97dc8f3e.txt
================================================
0 0.65601145 0.8740234375 0.7355279890624999 0.7470703125 0.7395038171875 0.6552734375 0.7792620859375 0.5869140625 0.799141221875 0.5205078125 0.8031170484375 0.3310546875 0.7474554703125 0.2021484375 0.6848361953125 0.134765625 0.6073075703125 0.1015625 0.511887721875 0.078125 0.4065283078125 0.0859375 0.2812897578125 0.1640625 0.196803434375 0.2939453125 0.1729484734375 0.3857421875 0.1868638671875 0.5732421875 0.2226463109375 0.6650390625 0.2305979640625 0.7587890625 0.2624045796875 0.8193359375 0.2892414125 0.845703125 0.309120546875 0.8515625 0.32899968125 0.880859375 0.384661259375 0.88671875 0.4144799625 0.912109375 0.4373409671875 0.9150390625 0.4701415390625 0.91015625 0.4880327609375 0.896484375 0.52779103125 0.912109375 0.65601145 0.8740234375


================================================
FILE: TumorDetection/train/labels/no_tumor_1470_jpg.rf.c24b57bab59623fe01a2eb0857080204.txt
================================================
0 0.806640625 0.5927734375 0.802734375 0.4306640625 0.740234375 0.2138671875 0.6904296875 0.154296875 0.6103515625 0.103515625 0.4150390625 0.09765625 0.3388671875 0.1328125 0.251953125 0.2509765625 0.203125 0.4794921875 0.21875 0.7060546875 0.3076171875 0.869140625 0.3681640625 0.90234375 0.55859375 0.9150390625 0.6572265625 0.89453125 0.740234375 0.8212890625 0.76953125 0.7666015625 0.806640625 0.5927734375


================================================
FILE: TumorDetection/train/labels/no_tumor_1473_jpg.rf.1888a952f485fcdd9a22c1a8d536464d.txt
================================================
0 0.9646920484375 0.5361328109375 0.9600652328124999 0.4013671890625 0.867528825 0.1982421890625 0.7854027625 0.11914062343750001 0.7183138640625 0.08007812343750001 0.5864494859375 0.044921876562500004 0.3666755140625 0.06445312343750001 0.2533184171875 0.10742187656249999 0.14111801875000002 0.1982421890625 0.0555218453125 0.3271484390625 0.0300743296875 0.4013671890625 0.050895025 0.5712890609375 0.10179004843749999 0.7470703109375 0.1688789421875 0.8408203109375 0.246378184375 0.90625 0.3296609515625 0.94921875 0.5089502421875001 0.9716796890625 0.6350310968749999 0.95703125 0.7819326453125 0.8935546890625 0.867528825 0.7880859390625 0.9646920484375 0.5361328109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1477_jpg.rf.e715b48cc76a3249b447c1d9bb61cefa.txt
================================================
0 0.7734375 0.7880859375 0.8164062515625 0.7099609375 0.8476562515625 0.5615234375 0.8476562515625 0.5029296875 0.8320312515625 0.4169921875 0.7929687484375 0.3076171875 0.755859375 0.2255859375 0.7070312515625 0.1552734375 0.6650390625 0.12109374843750001 0.6005859375 0.09375 0.5400390625 0.08203125156249999 0.4716796875 0.08203125156249999 0.3974609375 0.095703125 0.3369140625 0.12109374843750001 0.2734375 0.1826171875 0.236328125 0.2451171875 0.177734375 0.3974609375 0.1601562515625 0.4677734375 0.15625 0.5537109375 0.1835937484375 0.7099609375 0.224609375 0.7861328125 0.2695312515625 0.8388671875 0.3212890625 0.8828125 0.3837890625 0.9140625 0.4326171875 0.923828125 0.5078125 0.9248046875 0.5947265625 0.9179687484375 0.6494140625 0.896484375 0.7197265625 0.8476562515625 0.7734375 0.7880859375


================================================
FILE: TumorDetection/train/labels/no_tumor_1483_jpg.rf.3bebb7373057a2a28118160040aae5ab.txt
================================================
0 0.712890625 0.7158203125 0.755859375 0.6611328125 0.7773437484375 0.5869140625 0.7734375 0.4755859375 0.744140625 0.3642578125 0.734375 0.3017578125 0.708984375 0.2509765625 0.6806640625 0.2109375 0.6298828125 0.1796875 0.5791015625 0.1640625 0.5419921875 0.1640625 0.5205078125 0.1757812515625 0.4912109375 0.162109375 0.4580078125 0.1640625 0.3720703125 0.189453125 0.333984375 0.2275390625 0.306640625 0.2705078125 0.265625 0.3759765625 0.2578125 0.4169921875 0.228515625 0.4794921875 0.21875 0.5498046875 0.2226562515625 0.5869140625 0.265625 0.6943359375 0.3007812515625 0.7548828125 0.3349609375 0.794921875 0.3935546875 0.818359375 0.4267578125 0.8242187484375 0.4814453125 0.822265625 0.5322265625 0.833984375 0.5703125 0.8330078125 0.6181640625 0.818359375 0.6748046875 0.7734375 0.693359375 0.7548828125 0.712890625 0.7158203125


================================================
FILE: TumorDetection/train/labels/no_tumor_1493_jpg.rf.3d4d7b8079dcba963e761cc65fc20050.txt
================================================
0 0.734375 0.7333984359375 0.8027343734375 0.5205078140625 0.8085937515625 0.3642578140625 0.7597656265625 0.2763671859375 0.6689453109375 0.205078125 0.5673828140625 0.1621093734375 0.4345703109375 0.1640624984375 0.3115234359375 0.21875 0.2148437515625 0.3134765640625 0.1875 0.3994140625 0.1914062484375 0.5029296890625 0.3046875015625 0.7958984359375 0.3994140625 0.861328125 0.4638671859375 0.861328125 0.5048828140625 0.8398437515625 0.5507812484375 0.8681640625 0.6201171859375 0.8535156265625 0.6533203109375 0.8203124984375 0.6806640625 0.8183593734375 0.734375 0.7333984359375


================================================
FILE: TumorDetection/train/labels/no_tumor_1495_jpg.rf.239bffb0f85441405a6dca078ecfe55a.txt
================================================
0 0.787109375 0.3212890625 0.6943359375 0.1484375 0.5537109375 0.087890625 0.4287109375 0.0859375 0.3076171875 0.140625 0.2265625 0.2431640625 0.193359375 0.3916015625 0.1953125 0.5595703125 0.2421875 0.7392578125 0.296875 0.8310546875 0.3935546875 0.8984375 0.5390625 0.9091796875 0.6552734375 0.865234375 0.744140625 0.7548828125 0.796875 0.5302734375 0.787109375 0.3212890625


================================================
FILE: TumorDetection/train/labels/no_tumor_1508_jpg.rf.e6ab7615bdeff09399d095c91a83cdd2.txt
================================================
0 0.765625 0.8059807859375001 0.8105468765625 0.686576225 0.85546875 0.5054106843749999 0.86328125 0.4415910046875 0.8574218765625 0.32424514375 0.8261718765625 0.23572107187500002 0.81640625 0.17190139375 0.79296875 0.1224926109375 0.7548828125 0.07617187343750001 0.6806640625 0.014410896875 0.6318359375 0.0020586984375 0.3720703125 0.0041173984375 0.2939453125 0.05146748125 0.2128906234375 0.17190139375 0.1699218765625 0.316010346875 0.16796875 0.43541490781250003 0.17578125 0.47864759218749997 0.23828125 0.6906936234375001 0.27734375 0.7915698890625 0.30859375 0.845096071875 0.3505859375 0.8852407093749999 0.4189453125 0.9243559953125 0.4755859375 0.9387668921874999 0.5605468765625 0.9418549421875 0.6064453125 0.93464949375 0.6630859375 0.9099451015625 0.7099609375 0.8728885140625 0.765625 0.8059807859375001


================================================
FILE: TumorDetection/train/labels/no_tumor_1519_jpg.rf.5c6ac517d7c17eb648b83d17b19f5f66.txt
================================================
0 0.7734375 0.8251953125 0.8007812515625 0.7587890625 0.826171875 0.6650390625 0.8359375 0.5888671875 0.826171875 0.4755859375 0.78125 0.2763671875 0.748046875 0.2001953125 0.7060546875 0.1523437484375 0.6455078125 0.10546874843750001 0.5888671875 0.08203125156249999 0.5419921875 0.07421874843750001 0.4169921875 0.07421874843750001 0.3681640625 0.08984374843750001 0.3173828125 0.119140625 0.263671875 0.1689453125 0.2304687484375 0.2294921875 0.197265625 0.3251953125 0.171875 0.4599609375 0.166015625 0.5087890625 0.166015625 0.6044921875 0.171875 0.6474609375 0.189453125 0.7333984375 0.2109375 0.7939453125 0.2783203125 0.8828125 0.4013671875 0.947265625 0.4970703125 0.958984375 0.546875 0.9580078125 0.6025390625 0.951171875 0.7080078125 0.904296875 0.7304687484375 0.8857421875 0.7734375 0.8251953125


================================================
FILE: TumorDetection/train/labels/no_tumor_1532_jpg.rf.e4162f45c4274d225354927c0bf0bf41.txt
================================================
0 0.894191575 0.4423828109375 0.8575067921875 0.3232421890625 0.793308425 0.2080078109375 0.69586446875 0.119140625 0.5628821359375 0.052734375 0.372579821875 0.046875 0.2533542796875 0.09179687656249999 0.1513247296875 0.1845703125 0.048148778125 0.3876953125 0.0320991828125 0.5556640625 0.068783965625 0.7294921890625 0.1765455171875 0.8701171890625 0.34506623593749997 0.9492187484375 0.495244565625 0.9599609375 0.6660580828125 0.9101562515625 0.7841372265625 0.8251953125 0.866677990625 0.6748046875 0.894191575 0.4423828109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1546_jpg.rf.b78874b776c0b0dcccc9e8b6688a9342.txt
================================================
0 0.7910156265625 0.6259765640625 0.7890624984375 0.4931640625 0.7578124984375 0.3935546890625 0.7539062484375 0.2900390625 0.6796875015625 0.1787109375 0.5673828140625 0.125 0.5302734359375 0.125 0.5009765640625 0.154296875 0.4755859375 0.125 0.4345703109375 0.11718750156249999 0.3408203109375 0.1484375015625 0.2578124984375 0.2763671859375 0.2109375015625 0.4619140625 0.2109375015625 0.6064453109375 0.2578124984375 0.7451171859375 0.3095703109375 0.8203124984375 0.4169921859375 0.8964843734375 0.4619140625 0.9003906265625 0.4970703109375 0.8671875015625 0.5185546890625 0.904296875 0.544921875 0.9072265640625 0.6220703109375 0.8828124984375 0.6787109375 0.8359375015625 0.7578124984375 0.7236328140625 0.7910156265625 0.6259765640625


================================================
FILE: TumorDetection/train/labels/no_tumor_1556_jpg.rf.f1971f5978217035ed2789638b8338f1.txt
================================================
0 0.8605885921875001 0.6689453109375 0.8814092812500001 0.5439453109375 0.876782465625 0.3974609390625 0.8235740296875 0.3076171890625 0.8166337968749999 0.2470703109375 0.7090602234375 0.1484375 0.5864494859375 0.09960937656249999 0.373615746875 0.10742187656249999 0.270668990625 0.1455078109375 0.175819175 0.2763671890625 0.1202973296875 0.4189453109375 0.11567050937500001 0.6630859390625 0.1920130484375 0.7783203109375 0.315780490625 0.859375 0.4268241796875 0.8984375 0.5667854953125 0.9052734390625 0.6188372281250001 0.8964843765625 0.732194325 0.8378906234375 0.8050667468749999 0.7802734390625 0.8605885921875001 0.6689453109375


================================================
FILE: TumorDetection/train/labels/no_tumor_1559_jpg.rf.c32373b116b9af904e4a8bd66e7ea912.txt
================================================
0 0.7265625 0.8212890625 0.765625 0.7529296875 0.8203125 0.6162109375 0.8476562515625 0.4951171875 0.8476562515625 0.4306640625 0.8164062515625 0.2900390625 0.8046875 0.2646484375 0.7734375 0.2119140625 0.7080078125 0.142578125 0.6533203125 0.107421875 0.5986328125 0.083984375 0.5478515625 0.076171875 0.4423828125 0.076171875 0.3876953125 0.0859375 0.3408203125 0.10546874843750001 0.2802734375 0.150390625 0.2148437484375 0.2275390625 0.181640625 0.3017578125 0.1601562515625 0.4111328125 0.15625 0.4912109375 0.162109375 0.5380859375 0.1835937484375 0.6162109375 0.2382812515625 0.7587890625 0.2773437484375 0.8232421875 0.3251953125 0.8710937484375 0.3720703125 0.896484375 0.4599609375 0.9179687484375 0.509765625 0.9189453125 0.5576171875 0.916015625 0.6318359375 0.896484375 0.6923828125 0.859375 0.7265625 0.8212890625


================================================
FILE: TumorDetection/train/labels/no_tumor_1561_jpg.rf.505e7b45cfce48a858123334c9861cd9.txt
================================================
0 0.69404978125 0.8242187515625 0.7567587203125 0.7431640640625 0.8587936046875001 0.4990234359375 0.8630450609375 0.3662109359375 0.8035247062499999 0.2294921875 0.685546875 0.1367187515625 0.6005178046874999 0.1171875 0.451716934375 0.1171875 0.3518077765625 0.1523437515625 0.248710028125 0.2431640640625 0.1955668609375 0.3486328125 0.17856104687500002 0.4345703125 0.1955668609375 0.4404296875 0.1976925859375 0.5751953125 0.22107558125 0.6533203125 0.29760174375 0.7958984359375 0.445339753125 0.8710937515625 0.506985828125 0.875 0.5239916421874999 0.8613281234375 0.5335574140625 0.8779296875 0.49742005625 0.8916015640625 0.556940409375 0.8955078125 0.5952034890625 0.8837890640625 0.55375181875 0.8769531234375 0.6026435296875 0.8710937515625 0.69404978125 0.8242187515625


================================================
FILE: TumorDetection/train/labels/no_tumor_1565_jpg.rf.1dc6b0996167862af8d243634c16d21b.txt
================================================
0 0.38657785 0 0.23048332343750003 0.041015625 0.102437034375 0.1357421875 0.029267721875000003 0.3212890640625 0.0609744234375 0.6083984375 0.148777596875 0.7880859359375 0.281701840625 0.8867187484375 0.4231625078125 0.931640625 0.517063125 0.9365234375 0.6963279296875 0.90625 0.8682758109375 0.8076171875 0.94388409375 0.6943359359375 0.9926636375 0.5283203109375 0.9926636375 0.2939453109375 0.9414451187499999 0.1630859359375 0.82071575625 0.048828123437499996 0.662182253125 0 0.38657785 0


================================================
FILE: TumorDetection/train/labels/no_tumor_1569_jpg.rf.bf40473a0dff4ca83c9b478a28bdede1.txt
================================================
0 0.861328125 0.4462890625 0.859375 0.2939453109375 0.8027343734375 0.1708984359375 0.6533203109375 0.042968751562500004 0.5595703109375 0.011718751562500001 0.4150390625 0.011718751562500001 0.2783203109375 0.06640624843750001 0.1679687515625 0.1669921859375 0.125 0.3662109375 0.1367187515625 0.6318359375 0.1953124984375 0.8466796890625 0.2470703109375 0.921875 0.3408203109375 0.9609375015625 0.5664062484375 0.9697265640625 0.6708984359375 0.951171875 0.7539062484375 0.8779296890625 0.814453125 0.7060546890625 0.861328125 0.4462890625


================================================
FILE: TumorDetection/train/labels/no_tumor_156_jpg.rf.8e6c8f1ea47b6317d6dafd806860be0e.txt
================================================
0 0.833984375 0.5293418796875 0.791015625 0.3622911890625 0.6953125 0.1827116953125 0.6064453109375 0.106494815625 0.4580078109375 0.0751728109375 0.3544921890625 0.09814228124999999 0.2421875 0.1889760921875 0.13671875 0.422847059375 0.11328125 0.5836333546875 0.18359375 0.8300331249999999 0.2099609390625 0.8749279953125001 0.3564453109375 0.958453340625 0.490234375 0.9720262125 0.6298828109375 0.9396601375 0.7578125 0.838385659375 0.83203125 0.587809621875 0.833984375 0.5293418796875


================================================
FILE: TumorDetection/train/labels/no_tumor_1573_jpg.rf.b38a4104755fedebfa3b321f128d9ba2.txt
================================================
0 0.841796875 0.4189453109375 0.8085937515625 0.2509765640625 0.7539062484375 0.1396484359375 0.6923828140625 0.08593750156249999 0.5556640625 0.041015626562500004 0.3857421859375 0.0566406265625 0.2646484359375 0.1289062484375 0.1992187515625 0.2275390625 0.1640624984375 0.4013671859375 0.189453125 0.6474609375 0.2539062484375 0.8173828140625 0.3544921859375 0.904296875 0.4169921859375 0.9257812484375 0.4921875015625 0.9267578140625 0.6298828140625 0.904296875 0.7421875015625 0.8076171859375 0.796875 0.6806640625 0.841796875 0.4189453109375


================================================
FILE: TumorDetection/train/labels/no_tumor_167_jpg.rf.c78f406de3d2740c6a21a1c26b4d4e73.txt
================================================
0 0.8535156234375 0.72035845625 0.859375 0.653863828125 0.83203125 0.3818403578125 0.75390625 0.1904164296875 0.6767578109375 0.10679439843750001 0.6162109390625 0.0765695703125 0.5439453109375 0.066494625 0.4130859390625 0.082614534375 0.2910156234375 0.17026654375 0.24609375 0.266986 0.1953125 0.5410244609375 0.22265625 0.7687181859375001 0.2871093765625 0.8896175062499999 0.3876953109375 0.9631645921875001 0.4980468765625 0.98835195 0.6357421890625 0.9631645921875001 0.7373046890625 0.8986849546875 0.8222656234375 0.7868530812500001 0.8535156234375 0.72035845625


================================================
FILE: TumorDetection/train/labels/no_tumor_172_jpg.rf.2c442fd363c927d59e8aa8c598e69627.txt
================================================
0 0.35546875 0.49833075937500004 0.35546875 0.4855802921875 0.349609375 0.4834552140625 0.3642578125 0.4760174421875 0.365234375 0.46857966875 0.3505859375 0.4441412734375 0.3330078125 0.437766040625 0.3251953125 0.44201619531249997 0.302734375 0.42182795625 0.296875 0.39420194375 0.3076171875 0.3825140171875 0.3369140625 0.3740137046875 0.3583984375 0.35701308125 0.4365234375 0.3208867578125 0.5810546875 0.314511525 0.6142578125 0.3230118359375 0.6337890625 0.3400124578125 0.6630859375 0.3485127703125 0.71875 0.41120256718750003 0.71875 0.419702878125 0.748046875 0.44520381249999996 0.775390625 0.4898304484375 0.7919921875 0.507893609375 0.8232421875 0.5248942328124999 0.833984375 0.4877053703125 0.830078125 0.4792050578125 0.84375 0.466454590625 0.84375 0.3878267109375 0.8125 0.2751975828125 0.7890625 0.2241957140625 0.7587890625 0.18913192968749998 0.736328125 0.1838192359375 0.7275390625 0.155130684375 0.7119140625 0.1445052953125 0.6982421875 0.1445052953125 0.6923828125 0.1657560734375 0.681640625 0.14344275625 0.6650390625 0.127504671875 0.6484375 0.141317678125 0.6328125 0.1710687671875 0.6328125 0.1859443140625 0.6279296875 0.18700685156250002 0.6328125 0.1115665875 0.6259765625 0.1020037375 0.6083984375 0.099878659375 0.5927734375 0.11262912656249999 0.5361328125 0.1211294390625 0.4931640625 0.1190043609375 0.4814453125 0.1360049828125 0.4560546875 0.1232545171875 0.4072265625 0.1232545171875 0.3662109375 0.15300560624999998 0.3388671875 0.1636309953125 0.3115234375 0.18913192968749998 0.2841796875 0.204007475 0.2802734375 0.21250778750000002 0.2666015625 0.21250778750000002 0.2626953125 0.22100809843749997 0.2529296875 0.22100809843749997 0.2392578125 0.2358836421875 0.2373046875 0.2316334859375 0.21875 0.25607188281250004 0.21875 0.28794805156250003 0.2041015625 0.2890105890625 0.1875 0.309198828125 0.185546875 0.321949296875 0.171875 0.3261994515625 0.158203125 0.3580756203125 0.15625 0.4069524125 0.1640625 0.45370412499999996 0.171875 0.4600793578125 0.1748046875 0.4760174421875 0.1845703125 0.4717672859375 0.1904296875 0.4802675984375 0.203125 0.4813301359375 0.2060546875 0.516393921875 0.2099609375 0.50576853125 0.2216796875 0.50576853125 0.2333984375 0.516393921875 0.2470703125 0.507893609375 0.2841796875 0.5100186875 0.28515625 0.5025809156250001 0.2783203125 0.4972682203125 0.3095703125 0.48876790937500003 0.3212890625 0.49301806406250004 0.32421875 0.4855802921875 0.31640625 0.4813301359375 0.3271484375 0.4781425203125 0.328125 0.494080603125 0.3359375 0.49620568125000003 0.3349609375 0.5015183750000001 0.328125 0.5004558375 0.328125 0.525956771875 0.3427734375 0.5227691546875 0.34765625 0.5323320046875 0.369140625 0.521706615625 0.361328125 0.5153313828125 0.3671875 0.506831071875 0.35546875 0.49833075937500004


================================================
FILE: TumorDetection/train/labels/no_tumor_176_jpg.rf.211bf548ffa63887aed9bdc6bedaf2f9.txt
================================================
0 0.8320312484375 0.4326171859375 0.7851562484375 0.3095703109375 0.6630859375 0.1503906265625 0.5517578140625 0.10546875156249999 0.4423828140625 0.10546875156249999 0.3427734359375 0.1484375015625 0.201171875 0.3212890625 0.1601562484375 0.6220703109375 0.2128906265625 0.8095703109375 0.2626953109375 0.8847656265625 0.3720703109375 0.951171875 0.533203125 0.9697265640625 0.6943359375 0.9277343734375 0.810546875 0.8212890625 0.84375 0.6552734359375 0.8320312484375 0.4326171859375


================================================
FILE: TumorDetection/train/labels/no_tumor_17_jpg.rf.47d32fcac83806a34fc6a14db18749a2.txt
================================================
0 0.8554687515625 0.5517578140625 0.8496093734375 0.3603515640625 0.814453125 0.2646484359375 0.7216796890625 0.1484375015625 0.5888671859375 0.08203124843750001 0.4267578140625 0.08203124843750001 0.3076171859375 0.1347656265625 0.2246093734375 0.2275390625 0.1503906265625 0.4072265640625 0.1640624984375 0.6787109375 0.2246093734375 0.8017578140625 0.3408203109375 0.892578125 0.482421875 0.9189453109375 0.6337890625 0.9003906265625 0.7763671859375 0.8085937515625 0.826171875 0.7138671859375 0.8554687515625 0.5517578140625


================================================
FILE: TumorDetection/train/labels/no_tumor_18_jpg.rf.1351a2b7f858de523903135931f21c2a.txt
================================================
0 0.8242187515625 0.7431640625 0.8398437515625 0.4970703109375 0.8183593734375 0.3681640625 0.779296875 0.2607421859375 0.7109375015625 0.1552734359375 0.5791015640625 0.07226562656249999 0.4404296890625 0.06835937343750001 0.3173828140625 0.1367187515625 0.2304687515625 0.2451171859375 0.1660156265625 0.4052734359375 0.1601562484375 0.5849609375 0.201171875 0.7783203109375 0.2958984359375 0.8808593734375 0.3720703109375 0.9277343734375 0.546875 0.9443359375 0.6259765640625 0.9335937515625 0.7021484359375 0.8964843734375 0.8027343734375 0.7939453109375 0.8242187515625 0.7431640625


================================================
FILE: TumorDetection/train/labels/no_tumor_196_jpg.rf.1e91801e190a5630099c4f0b1712d055.txt
================================================
0 0.81598710625 0.8349609375 0.868773334375 0.7548828140625 0.903964153125 0.6572265625 0.919360134375 0.5087890625 0.9017647249999999 0.3857421859375 0.84238021875 0.2705078140625 0.7654003046875 0.1708984375 0.68512125 0.11718750156249999 0.6411327250000001 0.10156249843750001 0.55755453125 0.083984375 0.46517863281249994 0.083984375 0.3618056046875 0.10156249843750001 0.31121880156250004 0.12109375 0.2650308515625 0.150390625 0.2045466328125 0.2119140625 0.1693558140625 0.2646484375 0.1165695859375 0.3740234375 0.1011736046875 0.4404296875 0.1011736046875 0.5439453125 0.1165695859375 0.6669921859375 0.15395983124999998 0.7607421859375 0.21554376249999999 0.8505859375 0.26063200000000003 0.88671875 0.34640961875 0.92578125 0.4409849453125 0.955078125 0.519064575 0.9560546875 0.6235373156249999 0.94140625 0.6939189531250001 0.919921875 0.7599017390625 0.884765625 0.81598710625 0.8349609375
4 0.598243915625 0.4912109375 0.6180387515625 0.4130859375 0.609241046875 0.3525390625 0.590545925 0.341796875 0.5135660093750001 0.390625 0.5003694515625 0.369140625 0.42338953593750006 0.333984375 0.41349211874999997 0.3486328140625 0.41349211874999997 0.3876953125 0.442084659375 0.4580078140625 0.433286953125 0.5185546875 0.4024949875 0.5810546875 0.40029556250000004 0.6025390625 0.443184371875 0.60546875 0.49817002656250003 0.568359375 0.5245631390625001 0.53515625 0.5366599828124999 0.5732421859375 0.5927453500000001 0.625 0.6114404734375001 0.6279296875 0.6268364562500001 0.6142578140625 0.6268364562500001 0.5947265625 0.6026427671875 0.5341796875 0.598243915625 0.4912109375


================================================
FILE: TumorDetection/train/labels/no_tumor_208_jpg.rf.0854557a52f738f8a9a581ade19e84ab.txt
================================================
0 0.3828124984375 0.8529706390625 0.3955078109375 0.8353515609375 0.4326171890625 0.8933908781250001 0.4814453125 0.90168220625 0.4853515625 0.9120463703125001 0.5341796875 0.9037550390625 0.5576171890625 0.8871723812500001 0.5830078109375 0.8270602328125 0.609375 0.8488249734375 0.6123046875 0.8871723812500001 0.6826171890625 0.9037550390625 0.7314453125 0.8892452125 0.7978515625 0.8871723812500001 0.8271484375 0.90168220625 0.8779296875 0.895463709375 0.921875 0.7907856609375 0.9121093734375 0.718236515625 0.9296875015625 0.635323209375 0.92578125 0.43633127343749994 0.859375 0.3140341484375 0.7958984375 0.261176915625 0.7841796875 0.277759578125 0.7568359359375 0.2238659296875 0.7216796875 0.21350176406250002 0.6865234375 0.21972026093750002 0.6533203125 0.1658266125 0.5615234375 0.1741179421875 0.5068359359375 0.1450982875 0.4638671890625 0.1450982875 0.2939453125 0.1865549390625 0.2294921890625 0.21972026093750002 0.1455078109375 0.2901965734375 0.123046875 0.33268964218749997 0.08789062656249999 0.3596364671875 0.06640625 0.4156029484375 0.05078125 0.6063035546875 0.05859375 0.6705613640625 0.046875 0.7265278484375 0.064453125 0.7866399937499999 0.140625 0.9151556187500001 0.1611328109375 0.9286290328125 0.2285156265625 0.93173828125 0.2822265625 0.9265561999999999 0.3251953125 0.895463709375 0.3857421890625 0.8830267125000001 0.3828124984375 0.8529706390625


================================================
FILE: TumorDetection/train/labels/no_tumor_210_jpg.rf.fdb3c871e7fc31c638d1ae4622f6655f.txt
================================================
0 0.5968543046875 0.6220703125 0.5947226828125001 0.6025390625 0.571274834375 0.5693359375 0.5627483437499999 0.5322265625 0.5766038906250001 0.51171875 0.6128414734375001 0.490234375 0.6682636578125 0.486328125 0.700237996875 0.501953125 0.7087644875 0.515625 0.7300807125000001 0.515625 0.7311465234375001 0.5283203125 0.7418046359375 0.5341796875 0.7418046359375 0.5439453125 0.732212334375 0.544921875 0.7204884109374999 0.5615234375 0.7311465234375001 0.5654296875 0.732212334375 0.578125 0.7918977640625 0.580078125 0.81108236875 0.587890625 0.8345302140624999 0.587890625 0.8611754968750001 0.5712890625 0.8697019875 0.5576171875 0.87396523125 0.4794921875 0.8697019875 0.4736328125 0.8718336093750001 0.4404296875 0.8483857609375001 0.3798828125 0.8569122515625001 0.3544921875 0.8569122515625001 0.3037109375 0.8313327812500001 0.2431640625 0.8100165562500001 0.2197265625 0.7705815390625 0.185546875 0.7364755781250001 0.169921875 0.7087644875 0.14453125 0.6831850171875 0.12890625 0.6426841875 0.1171875 0.60644660625 0.09765625 0.5915252484375 0.095703125 0.5616825328124999 0.08203125 0.5190500828125 0.080078125 0.5051945359375 0.0927734375 0.50945778125 0.1181640625 0.500931290625 0.1318359375 0.500931290625 0.1767578125 0.4934706125 0.1875 0.4817466890625 0.1630859375 0.4838783109375 0.1103515625 0.475351821875 0.0849609375 0.44870654062500004 0.078125 0.389021109375 0.0859375 0.3293356796875 0.103515625 0.27391349374999996 0.12890625 0.20996481875 0.1640625 0.1726614234375 0.1962890625 0.136423840625 0.2568359375 0.1193708609375 0.3017578125 0.1193708609375 0.3330078125 0.12789735156249998 0.3603515625 0.1385554640625 0.3759765625 0.1385554640625 0.3916015625 0.12789735156249998 0.3974609375 0.1193708609375 0.4150390625 0.1193708609375 0.4580078125 0.11084437031250001 0.4755859375 0.11084437031250001 0.4990234375 0.1321605953125 0.5439453125 0.1492135765625 0.5556640625 0.1492135765625 0.5693359375 0.15667425468750001 0.578125 0.20783319531250002 0.595703125 0.244070778125 0.59765625 0.27391349374999996 0.58203125 0.28883485000000003 0.583984375 0.29096647343750004 0.578125 0.3293356796875 0.576171875 0.3325331125 0.5654296875 0.341059603125 0.5615234375 0.341059603125 0.5498046875 0.3495860921875 0.5458984375 0.341059603125 0.5400390625 0.3495860921875 0.5302734375 0.339993790625 0.525390625 0.333598925 0.52734375 0.3293356796875 0.51953125 0.3229408109375 0.529296875 0.3080194546875 0.529296875 0.3069536421875 0.5166015625 0.29309809531250003 0.515625 0.2877690390625 0.5068359375 0.2899006625 0.4970703125 0.3016245859375 0.486328125 0.324006621875 0.4755859375 0.3261382453125 0.4541015625 0.3378621703125 0.4453125 0.343191225 0.4482421875 0.3495860921875 0.4697265625 0.37623137500000003 0.486328125 0.41779801250000004 0.5283203125 0.41566639062500005 0.5654296875 0.40287665625 0.5732421875 0.40500827812500007 0.5810546875 0.3986134109375 0.5888671875 0.40713990000000005 0.5966796875 0.40074503281249996 0.6025390625 0.40500827812500007 0.6123046875 0.40074503281249996 0.6201171875 0.41566639062500005 0.6298828125 0.40927152343750006 0.6337890625 0.4231270703125 0.640625 0.44657491718750003 0.642578125 0.4391142390625 0.6494140625 0.44444329531250004 0.654296875 0.4657595203125 0.65625 0.4742860109375 0.650390625 0.4806808765625 0.654296875 0.49133899062499997 0.6484375 0.49986548124999997 0.65625 0.5329056296875 0.6591796875 0.565945778125 0.65234375 0.5787355125 0.640625 0.592591059375 0.6357421875 0.5968543046875 0.6220703125


================================================
FILE: TumorDetection/train/labels/no_tumor_21_jpg.rf.68d32f400194ee91591f670f61f75e97.txt
================================================
0 0.897190825 0.7138671875 0.897190825 0.5087890625 0.8359707453125 0.3486328125 0.8254155578125 0.2705078125 0.7335854390625001 0.1464843734375 0.5984790546875 0.0859375 0.5309258640625 0.07226562656249999 0.49503823125 0.07617187343750001 0.47181682187499996 0.06445312656249999 0.43170711562499997 0.06445312656249999 0.381042221875 0.07226562656249999 0.2818234703125 0.10351562656249999 0.208992684375 0.1494140625 0.1456615703125 0.2333984375 0.126662234375 0.2685546875 0.122440159375 0.3291015625 0.1097739359375 0.3662109375 0.0928856390625 0.3876953125 0.06122008125 0.4619140625 0.048553854687499995 0.5185546875 0.0443317828125 0.5966796875 0.0591090421875 0.7294921875 0.07599734062499999 0.7783203125 0.1456615703125 0.8935546875 0.2206033921875 0.9609375 0.2818234703125 0.99609375 0.41692985312499997 0.9941406265625 0.4612616359375 0.9765625 0.48026097031250004 1 0.7156416218749999 0.9990234375 0.7620844421874999 0.9580078125 0.8127493359375 0.8974609375 0.8676363015625 0.8017578125 0.897190825 0.7138671875


================================================
FILE: TumorDetection/train/labels/no_tumor_222_jpg.rf.7cfb27d35a17497886dfe1ac98b66b0a.txt
================================================
0 0.6650390625 0.814453125 0.755859375 0.7255859375 0.796875 0.6416015625 0.8125 0.5595703125 0.810546875 0.5029296875 0.802734375 0.4716796875 0.7578125 0.3818359375 0.751953125 0.2841796875 0.720703125 0.2275390625 0.6611328125 0.171875 0.6396484375 0.1601562515625 0.5166015625 0.142578125 0.4990234375 0.1679687484375 0.4833984375 0.142578125 0.4638671875 0.140625 0.4052734375 0.1484375 0.3330078125 0.1796875 0.291015625 0.2216796875 0.2539062515625 0.2744140625 0.244140625 0.3154296875 0.2421875 0.3857421875 0.1914062515625 0.4931640625 0.1875 0.5458984375 0.1953125 0.5693359375 0.197265625 0.6142578125 0.216796875 0.6572265625 0.2226562515625 0.6884765625 0.2578125 0.7392578125 0.3017578125 0.791015625 0.3681640625 0.8359375 0.4072265625 0.8515625 0.453125 0.8564453125 0.4755859375 0.833984375 0.4912109375 0.8515625 0.5673828125 0.8515625 0.6650390625 0.814453125


================================================
FILE: TumorDetection/train/labels/no_tumor_227_jpg.rf.2630c3fc56c36439fdbd417a8a5f80cd.txt
================================================
0 0.8359375 0.7607421875 0.8632812484375 0.6416015640625 0.8632812484375 0.5400390640625 0.833984375 0.3681640640625 0.8007812484375 0.2958984359375 0.736328125 0.2021484359375 0.6923828125 0.1679687515625 0.5947265640625 0.12109375156249999 0.4599609359375 0.099609375 0.3876953125 0.12109375156249999 0.2919921875 0.1757812484375 0.197265625 0.2783203125 0.1679687515625 0.3525390640625 0.1367187515625 0.5048828125 0.1328125 0.6103515640625 0.140625 0.6962890640625 0.181640625 0.7861328125 0.2412109359375 0.849609375 0.2939453125 0.890625 0.3896484359375 0.927734375 0.5742187515625 0.9306640640625 0.6142578125 0.9257812484375 0.6923828125 0.892578125 0.8164062484375 0.7919921875 0.8359375 0.7607421875


================================================
FILE: TumorDetection/train/labels/no_tumor_239_jpg.rf.995bb1e62cc872f5b99f60049f4538ea.txt
================================================
0 0.9573384828125 0.7158203109375 0.9861306187500001 0.6220703109375 0.9741338953125 0.3701171890625 0.8997542140625001 0.1806640609375 0.8421699437499999 0.11230468906249999 0.7689899328124999 0.05859375 0.6970095953125 0.03515625 0.5650456484374999 0.015625 0.32271184218750004 0.029296875 0.226738059375 0.05859375 0.14396067343750002 0.1337890609375 0.1055711609375 0.1923828109375 0.055184924999999996 0.3095703109375 0.026392790625000002 0.4482421890625 0.028792134375 0.6376953109375 0.09117509375 0.7841796890625 0.21714068125 0.904296875 0.37549742343750003 0.958984375 0.5350538390625 0.9638671890625 0.6226299171874999 0.958984375 0.7569932140625 0.921875 0.8997542140625001 0.8212890609375 0.9573384828125 0.7158203109375


================================================
FILE: TumorDetection/train/labels/no_tumor_274_jpg.rf.f693719103f198d85158ae961580ca22.txt
================================================
0 0.9447274234374999 0.6865234359375 0.9808645374999999 0.5498046859375 0.942146203125 0.3681640640625 0.8363160796875 0.1806640640625 0.721451678125 0.09960937343750001 0.571740775 0.06445312656249999 0.4452608765625 0.06640625 0.36008053281249996 0.08203125 0.2297288015625 0.1513671859375 0.0774366734375 0.3525390640625 0.0232310046875 0.4970703140625 0.036137114062499996 0.6650390640625 0.10841134375000001 0.7998046859375 0.2129508546875 0.87109375 0.3187809734375 0.9199218734375 0.51624449375 0.9423828140625 0.63110889375 0.93359375 0.690477009375 0.9140625 0.84922219375 0.8154296859375 0.9447274234374999 0.6865234359375


================================================
FILE: TumorDetection/train/labels/no_tumor_283_jpg.rf.c1f48ded5592810a26b5ac0c952bcdd0.txt
================================================
0 0.8286830359374999 0.5947265625 0.8307756703125 0.4306640625 0.7742745531249999 0.2451171875 0.7177734359375 0.1591796875 0.6602260046875 0.11328125 0.62255859375 0.09570312656249999 0.60791015625 0.09765625 0.5974469859375 0.0859375 0.54931640625 0.07617187343750001 0.45724051406250005 0.07617187343750001 0.43108258906249997 0.0830078125 0.44049944218750003 0.09765625 0.44677734375 0.08398437343750001 0.45305524531249997 0.08789062656249999 0.45514787968750003 0.1015625 0.41538783593749995 0.10546875 0.40283203125 0.11523437343750001 0.40283203125 0.10742187343750001 0.42898995624999997 0.0947265625 0.40701729999999997 0.08984375 0.35888671875 0.11328125 0.315987721875 0.1494140625 0.257393971875 0.2236328125 0.25530133906249997 0.2548828125 0.2521623890625 0.2597656265625 0.24693080312500001 0.2548828125 0.24693080312500001 0.2666015625 0.2239118296875 0.2919921875 0.198800221875 0.3525390625 0.1715959828125 0.4775390625 0.17578125 0.5576171875 0.18415178593749998 0.5751953125 0.19461495625 0.6572265625 0.2615792421875 0.7861328125 0.31494140625 0.84375 0.3463309140625 0.8515625 0.3923688609375 0.88671875 0.44049944218750003 0.8886718734375 0.47398158593749995 0.87109375 0.48025948593749995 0.8828125 0.5074637281250001 0.8886718734375 0.5158342640625 0.8984375 0.556640625 0.8994140625 0.6455775671875 0.8691406265625 0.7146344859375 0.828125 0.7805524562499999 0.7333984375 0.8286830359374999 0.5947265625


================================================
FILE: TumorDetection/train/labels/no_tumor_296_jpg.rf.37d6dc5e9f6875ae8972c9433efdc70a.txt
================================================
0 0.865234375 0.3628522171875 0.8242187515625 0.2408677109375 0.7431640625 0.1426598453125 0.6123046875 0.0620260203125 0.5302734390625 0.049620815624999995 0.4306640625 0.057890954687499996 0.2880859375 0.1261195734375 0.1992187515625 0.2305300421875 0.1679687515625 0.342176875 0.146484375 0.67504985 0.166015625 0.7308732671875 0.109375 0.8094395609375 0.1181640625 0.8270135984375001 0.1484375015625 0.8280473640625001 0.150390625 0.9107487234375 0.1738281265625 0.9872474812500001 0.8515624984375 0.9934500843749999 0.890625 0.8053044890625 0.8925781265625 0.7039953265625 0.875 0.6295641 0.865234375 0.3628522171875


================================================
FILE: TumorDetection/train/labels/no_tumor_303_jpg.rf.f1d183452b4a78c0253cb066fccb1a73.txt
================================================
0 0.9941077593750001 0.6767578125 0.9964036203125 0.6162109359375 0.9872201781250001 0.3095703125 0.955078125 0.2314453125 0.9160484906249999 0.1689453125 0.8483205937499999 0.10546875 0.8001075171875 0.07421875 0.70827308125 0.039062498437499996 0.6256220890625 0.02734375 0.5360835140625 0.023437501562499997 0.44654494062500005 0.033203125 0.3363436171875 0.06640625 0.269763653125 0.10156249843750001 0.19514817499999998 0.1591796875 0.13775165312499998 0.2294921875 0.02525446875 0.4638671875 0.0114793046875 0.6044921875 0.0183668875 0.6474609359375 0.0413254953125 0.7177734375 0.11249718281250001 0.8408203125 0.23991746093750002 0.96875 0.3225684546875 1 0.7140127328125 0.9990234375 0.817326471875 0.9267578125 0.9160484906249999 0.8251953125 0.9780367328124999 0.7236328125 0.9941077593750001 0.6767578125


================================================
FILE: TumorDetection/train/labels/no_tumor_306_jpg.rf.a0b6f5a007474d22a0c7825b30c12c6c.txt
================================================
0 0.8184475203125 0.8876953125 0.9266719859375 0.7392578125 0.9537281015625 0.6787109375 0.9672561609375 0.5908203125 0.965001484375 0.4755859375 0.9537281015625 0.4072265625 0.89961586875 0.2431640625 0.8432489593750001 0.1494140625 0.77222665625 0.080078125 0.706841040625 0.044921875 0.654983484375 0.025390625 0.5760698125 0.009765625 0.4633359921875 0.019531251562499997 0.350602175 0.041015625 0.2401230328125 0.09375 0.200666196875 0.1318359375 0.121752521875 0.2294921875 0.09469640781250001 0.2880859375 0.0586215859375 0.4306640625 0.04283885 0.5712890625 0.049602878125 0.6591796875 0.07891367187500001 0.7392578125 0.11498849375 0.7958984375 0.2063028890625 0.890625 0.303253971875 0.9492187484375 0.3573662046875 0.970703125 0.5343583 0.9873046875 0.6098899578125 0.9804687484375 0.7090957171875 0.951171875 0.778990684375 0.9179687484375 0.8184475203125 0.8876953125


================================================
FILE: TumorDetection/train/labels/no_tumor_30_jpg.rf.df215710a36490ef52b7ddfaf254c1c3.txt
================================================
0 0.8867241890625 0.6669921875 0.9038235546875001 0.5908203125 0.89649525625 0.4638671875 0.8232122640625 0.2529296875 0.78657076875 0.2001953125 0.7169519281250001 0.13671875 0.5996991421875 0.107421875 0.5606148796875 0.109375 0.5215306187500001 0.125 0.4922174234375 0.10546875 0.41404889999999994 0.109375 0.287025046875 0.1503906265625 0.1905357765625 0.2666015640625 0.17832194375 0.3408203125 0.1245810828125 0.4248046875 0.11725278593749999 0.6025390625 0.14900875 0.6767578125 0.2051923734375 0.7607421875 0.33832314218750004 0.8671875015625 0.39694953437499997 0.8964843734375 0.4384765625 0.8984375015625 0.46168284374999996 0.8837890625 0.46168284374999996 0.8583984359375 0.48488912187500005 0.8496093734375 0.50076710625 0.8876953125 0.519087853125 0.8984375015625 0.586263928125 0.9072265640625 0.6925242640625 0.8671875015625 0.738936825 0.83984375 0.8500826968749999 0.7255859375 0.8867241890625 0.6669921875


================================================
FILE: TumorDetection/train/labels/no_tumor_310_jpg.rf.a8f3513a653d2a692f87c5d8258e3f1c.txt
================================================
0 0.8690878375000001 0.6416015640625 0.866870775 0.5244140640625 0.7715371625 0.2626953140625 0.7150021125 0.2011718734375 0.613017315625 0.1542968734375 0.5420713687500001 0.1503906265625 0.5065983968750001 0.18359375 0.4799936671875 0.1542968734375 0.4046135984375 0.1503906265625 0.29597761718750004 0.19140625 0.2061866578125 0.2939453140625 0.1418918921875 0.4833984359375 0.1241554046875 0.6376953140625 0.168496621875 0.7294921859375 0.29597761718750004 0.8417968734375 0.4068306578125 0.8886718734375 0.5786528734374999 0.8974609359375 0.6440561671875 0.8808593734375 0.757126265625 0.8203125 0.8225295578124999 0.7529296859375 0.8690878375000001 0.6416015640625


================================================
FILE: TumorDetection/train/labels/no_tumor_327_jpg.rf.e1977eb52d77e222443eeb2210f3c866.txt
================================================
0 0.8933395125000001 0.5537109390625 0.847805215625 0.3251953125 0.771914725 0.1845703125 0.7209596781250001 0.1367187484375 0.6559106859375 0.10351562656249999 0.5691786953125 0.08203125156249999 0.44992220781250003 0.08007812656249999 0.31765591875 0.1171875 0.2211665765625 0.1943359390625 0.15178098281250002 0.3212890609375 0.1127515890625 0.4326171875 0.10191008906250001 0.5712890609375 0.1626224828125 0.7666015609375 0.2341763765625 0.8740234390625 0.3935464109375 0.9628906265625 0.5290651484375 0.9716796875 0.6624155843749999 0.9414062515625 0.776251321875 0.8525390609375 0.85431011875 0.7255859390625 0.8933395125000001 0.5537109390625


================================================
FILE: TumorDetection/train/labels/no_tumor_333_jpg.rf.fccfc15a10aea59ee0201b416b5fbc54.txt
================================================
0 0.8007812484375 0.5567659375 0.7539062484375 0.333254115625 0.6757812484375 0.1862598546875 0.5693359359375 0.1067218625 0.5029296859375 0.1167899609375 0.4306640640625 0.092626521875 0.3837890640625 0.1067218625 0.3564453140625 0.14900788125 0.3134765625 0.16108960156250002 0.2265624984375 0.284927234375 0.1796875015625 0.4158125359375 0.158203125 0.5950247171874999 0.2109375015625 0.7400053578125 0.3115234375 0.8457204078125 0.3681640640625 0.87189746875 0.4960937515625 0.8910268578125 0.6318359359375 0.8779383281250001 0.7265624984375 0.8124956765625001 0.796875 0.669528659375 0.8007812484375 0.5567659375


================================================
FILE: TumorDetection/train/labels/no_tumor_349_jpg.rf.7b928feaaaf67a568893890d8a32e000.txt
================================================
0 0.8467293437500001 0.4912109359375 0.7904197546875 0.3134765640625 0.7111692281249999 0.2021484359375 0.610020525 0.13671875 0.47237486874999995 0.1171875 0.4035520375 0.1308593734375 0.33055813125 0.1699218734375 0.2335805078125 0.2763671859375 0.1710143015625 0.4287109359375 0.164757678125 0.5576171859375 0.229409428125 0.7412109359375 0.30136056718750004 0.8203125 0.3952098765625 0.8730468734375 0.55266816875 0.8818359359375 0.68092889375 0.8457031265625 0.7779065171874999 0.7626953140625 0.8300450218750001 0.6357421859375 0.8467293437500001 0.4912109359375


================================================
FILE: TumorDetection/train/labels/no_tumor_374_jpg.rf.45c3f5586a1f96c321c3e6a6be475732.txt
================================================
0 0.9070644937499999 0.6884765625 0.8863447484375 0.4990234375 0.82878989375 0.3095703140625 0.7297955453125 0.1376953140625 0.5790018296875 0.07031249843750001 0.519144778125 0.07031249843750001 0.49151845156249996 0.09765624843750001 0.4684965109375 0.07031249843750001 0.40633726875000004 0.068359375 0.29352975312499996 0.119140625 0.21640625156249998 0.1884765625 0.17726894843750002 0.2509765625 0.087483378125 0.4951171875 0.0644614390625 0.6767578125 0.128922871875 0.7958984375 0.2129529578125 0.8671875015625 0.344178025 0.9238281265625 0.45007895781249996 0.9394531265625 0.48461186874999995 0.9101562484375 0.521446975 0.9492187515625 0.5916638984375 0.9599609390625 0.7240400593749999 0.921875 0.810372340625 0.8603515625 0.8610206125 0.7880859390625 0.9070644937499999 0.6884765625


================================================
FILE: TumorDetection/train/labels/no_tumor_380_jpg.rf.fe6399bad33fea5225c6bd50d4c81c70.txt
================================================
0 0.9490691484375 0.1650390609375 0.8507147609375 0.06445312343750001 0.720927528125 0.0253906234375 0.6600897609375 0.0546875 0.658061834375 0.041015623437499996 0.633726728125 0.041015623437499996 0.4512134296875 0.06445312343750001 0.2970910890625 0.1269531234375 0.12573138281250001 0.2861328109375 0.0892287234375 0.3583984390625 0.0892287234375 0.4345703109375 0.16223404218749998 0.5908203109375 0.228141621875 0.640625 0.4289062515625 0.6542968765625 0.47960438749999995 0.69921875 0.4998836421875 0.6855468765625 0.5120511953125 0.69921875 0.49481382968750004 0.6748046890625 0.5171210125 0.6533203109375 0.462367021875 0.5830078109375 0.5120511953125 0.5644531234375 0.5343583781250001 0.5722656234375 0.5505817828125 0.546875 0.5860704765625 0.6669921890625 0.6540059828125 0.73828125 0.6752992 0.7431640609375 0.6834109015625 0.6435546890625 0.7138297875 0.6435546890625 0.7219414890625 0.7177734390625 0.7848071828125 0.8583984390625 0.93385970625 0.7675781234375 0.9632646296875 0.7333984390625 0.9632646296875 0.2392578109375 0.9490691484375 0.1650390609375


================================================
FILE: TumorDetection/train/labels/no_tumor_389_jpg.rf.eb17318825b175228ac172bb0919a71b.txt
================================================
0 0.6462265203124999 0.8125 0.6826336453125 0.810546875 0.6930356828125 0.798828125 0.7242417921875 0.7929687515625 0.7593486656249999 0.7626953125 0.808758340625 0.6982421875 0.8165598656249999 0.6533203125 0.8269619046875001 0.6513671875 0.821760884375 0.6455078125 0.8633690312500001 0.5888671875 0.8815725953125 0.5087890640625 0.8815725953125 0.4462890640625 0.8477659781250001 0.3525390640625 0.7398448484375 0.2226562484375 0.6618295734374999 0.193359375 0.6527277921875 0.1865234359375 0.6618295734374999 0.181640625 0.6436260078125 0.181640625 0.609819390625 0.162109375 0.537005134375 0.15625 0.5396056437500001 0.142578125 0.5188015703125 0.1484375 0.5110000453125 0.1367187515625 0.505799025 0.1484375 0.4927964796875 0.1367187515625 0.48239444374999996 0.1445312484375 0.4693918984375 0.134765625 0.4563893515625 0.140625 0.41998222500000004 0.1367187515625 0.40437916874999996 0.140625 0.39397713125 0.15625 0.3237633875 0.1796875 0.28995676718749996 0.201171875 0.2821552421875 0.1953125 0.2665521859375 0.2109375 0.252249384375 0.2060546875 0.2574504046875 0.2177734359375 0.1534300390625 0.3369140640625 0.132625965625 0.4072265640625 0.12482443750000001 0.4072265640625 0.11182189375 0.4892578125 0.132625965625 0.5341796875 0.127424946875 0.5419921875 0.143028003125 0.5556640640625 0.14822902031250001 0.5908203125 0.1872366578125 0.6455078125 0.1872366578125 0.6689453125 0.205440221875 0.6884765640625 0.21584225781250002 0.7275390640625 0.252249384375 0.7626953125 0.27045295 0.7685546875 0.2730534578125 0.7880859359375 0.36537153437500003 0.8203125 0.4407862984375 0.830078125 0.4459873140625 0.837890625 0.514900809375 0.8408203125 0.570811753125 0.8398437515625 0.5838143 0.828125 0.6176209187499999 0.828125 0.6462265203124999 0.8125


================================================
FILE: TumorDetection/train/labels/no_tumor_395_jpg.rf.8611304494e5f48b2b5bdee1681f17d0.txt
================================================
0 0.728515625 0.8212890625 0.779296875 0.7275390625 0.828125 0.5966796875 0.8476562515625 0.4970703125 0.8476562515625 0.4306640625 0.8398437484375 0.3798828125 0.8085937484375 0.2685546875 0.7617187484375 0.1943359375 0.7080078125 0.140625 0.6533203125 0.10546874843750001 0.6044921875 0.083984375 0.5458984375 0.07421874843750001 0.4482421875 0.07421874843750001 0.3837890625 0.0859375 0.3212890625 0.1171875 0.2861328125 0.1445312515625 0.224609375 0.2119140625 0.1796875 0.3017578125 0.158203125 0.4189453125 0.15625 0.4990234375 0.171875 0.5830078125 0.2304687484375 0.7451171875 0.2851562515625 0.8349609375 0.3212890625 0.869140625 0.3662109375 0.8945312515625 0.4462890625 0.916015625 0.513671875 0.9189453125 0.5869140625 0.9101562515625 0.6494140625 0.888671875 0.6943359375 0.859375 0.728515625 0.8212890625


================================================
FILE: TumorDetection/train/labels/no_tumor_400_jpg.rf.9ef016ab9dfe63e92695bcb66c66c089.txt
================================================
0 0.8933203093750001 0.5537109359375 0.922851559375 0.4833984359375 0.9253125000000001 0.3212890640625 0.8514843750000001 0.2021484359375 0.7001367203125 0.07226562656249999 0.6238476546875 0.05078125 0.3654492203125 0.0566406265625 0.2104101546875 0.1347656265625 0.118125 0.2451171859375 0.06890625 0.3583984359375 0.07875 0.5205078140625 0.105820309375 0.6376953140625 0.1575 0.7041015640625 0.20671875 0.8212890640625 0.34576172031249996 0.90625 0.55125 0.9287109359375 0.7468945296875 0.87890625 0.824414059375 0.7939453140625 0.868710940625 0.7119140640625 0.8933203093750001 0.5537109359375


================================================
FILE: TumorDetection/train/labels/no_tumor_405_jpg.rf.615ee173a7516905e28a71c8075814a2.txt
================================================
0 0.9926636375 0.4794921875 0.9609569359375 0.3525390640625 0.885348646875 0.2099609359375 0.7646192890625 0.125 0.6012078265625 0.07031250156249999 0.40121171718750004 0.07421874843750001 0.2182884421875 0.150390625 0.148777596875 0.2119140640625 0.112192940625 0.2724609359375 0.063413403125 0.3857421875 0.0341456765625 0.5283203109375 0.024389771875 0.6552734375 0.056096471875 0.7744140640625 0.11707089531249999 0.8818359359375 0.2695069546875 0.978515625 0.34267626875 1 0.6999863984375 0.9990234375 0.81827678125 0.953125 0.9024214875000001 0.8837890640625 0.9804687515625 0.7529296890625 0.9926636375 0.4794921875


================================================
FILE: TumorDetection/train/labels/no_tumor_406_jpg.rf.7f1557f0e11ab67ae8859c327f09cf52.txt
================================================
0 0.7734375 0.7861328125 0.818359375 0.6982421875 0.8398437484375 0.5849609375 0.84375 0.5087890625 0.8203125 0.3955078125 0.7617187484375 0.2431640625 0.7265625 0.1826171875 0.6992187484375 0.1513671875 0.6591796875 0.119140625 0.6142578125 0.099609375 0.5322265625 0.08203125156249999 0.4638671875 0.08203125156249999 0.3994140625 0.09375 0.3349609375 0.12109374843750001 0.3076171875 0.140625 0.271484375 0.1806640625 0.2148437484375 0.2880859375 0.177734375 0.3857421875 0.154296875 0.4931640625 0.1523437484375 0.5595703125 0.1679687484375 0.6474609375 0.1992187484375 0.7431640625 0.2265625 0.7880859375 0.2939453125 0.859375 0.3564453125 0.8984375 0.4013671875 0.916015625 0.4501953125 0.923828125 0.529296875 0.9248046875 0.6044921875 0.916015625 0.6591796875 0.8945312515625 0.7080078125 0.859375 0.7734375 0.7861328125


================================================
FILE: TumorDetection/train/labels/no_tumor_411_jpg.rf.8fb8cd910f986e91aca953f4badb7285.txt
================================================
0 0.6796875015625 0.7998046890625 0.6796875015625 0.7919921859375 0.6953124984375 0.7880859375 0.6953124984375 0.7822265640625 0.6904296890625 0.7851562484375 0.6875 0.7763671859375 0.6875 0.7353515640625 0.6962890625 0.732421875 0.6992187515625 0.7412109375 0.703125 0.7412109375 0.703125 0.7041015640625 0.7109375015625 0.7021484359375 0.7119140625 0.6933593734375 0.71875 0.6962890625 0.71875 0.6572265640625 0.7265624984375 0.6552734359375 0.7265624984375 0.6455078140625 0.716796875 0.6416015640625 0.7177734359375 0.6289062484375 0.7070312484375 0.6298828140625 0.7070312484375 0.6396484359375 0.7148437515625 0.6435546890625 0.7080078140625 0.6445312484375 0.7060546890625 0.6523437515625 0.7021484359375 0.6445312484375 0.6923828140625 0.6445312484375 0.6933593734375 0.6591796890625 0.6757812484375 0.6611328140625 0.6748046890625 0.6757812484375 0.671875 0.6748046890625 0.673828125 0.6630859375 0.6640624984375 0.6572265640625 0.6865234359375 0.65625 0.6875 0.6455078140625 0.6728515640625 0.6464843734375 0.669921875 0.6416015640625 0.6728515640625 0.638671875 0.7021484359375 0.640625 0.7021484359375 0.6367187515625 0.6865234359375 0.6367187515625 0.6845703109375 0.626953125 0.6767578140625 0.6289062484375 0.6748046890625 0.6367187515625 0.6708984359375 0.6289062484375 0.6611328140625 0.6289062484375 0.6621093734375 0.6435546890625 0.6494140625 0.6464843734375 0.6484375015625 0.6416015640625 0.65625 0.6396484359375 0.65625 0.6298828140625 0.6455078140625 0.6289062484375 0.6445312484375 0.6513671859375 0.640625 0.6513671859375 0.6396484359375 0.6445312484375 0.6318359375 0.6445312484375 0.6279296890625 0.6621093734375 0.623046875 0.6435546890625 0.625 0.6298828140625 0.6220703109375 0.626953125 0.6201171859375 0.6308593734375 0.6132812484375 0.6298828140625 0.6132812484375 0.7236328140625 0.6240234359375 0.734375 0.625 0.7197265640625 0.6289062484375 0.7197265640625 0.6289062484375 0.7392578140625 0.6367187515625 0.7509765640625 0.6318359375 0.7578124984375 0.6396484359375 0.7597656265625 0.640625 0.7509765640625 0.6445312484375 0.7509765640625 0.6464843734375 0.8115234359375 0.6494140625 0.8046875015625 0.6503906265625 0.8193359375 0.65625 0.8193359375 0.6572265640625 0.8046875015625 0.6865234359375 0.8046875015625 0.6845703109375 0.796875 0.6796875015625 0.7998046890625
0 0.6923828140625 0.609375 0.7421875015625 0.6083984359375 0.7421875015625 0.5986328140625 0.734375 0.5947265640625 0.75 0.5927734359375 0.7529296890625 0.4882812484375 0.7558593734375 0.5419921859375 0.7626953109375 0.5566406265625 0.767578125 0.5517578140625 0.765625 0.4853515640625 0.751953125 0.4755859375 0.7539062484375 0.4658203109375 0.7607421859375 0.4648437515625 0.7646484359375 0.4804687515625 0.765625 0.4619140625 0.751953125 0.4619140625 0.748046875 0.4228515640625 0.7451171859375 0.439453125 0.734375 0.4072265640625 0.7177734359375 0.390625 0.7128906265625 0.3955078140625 0.7109375015625 0.3837890625 0.7177734359375 0.3847656265625 0.71875 0.3759765640625 0.7089843734375 0.3740234359375 0.7109375015625 0.3525390625 0.7041015640625 0.3691406265625 0.6953124984375 0.3486328140625 0.6816406265625 0.3447265640625 0.6787109375 0.3125 0.6757812484375 0.3818359375 0.6708984359375 0.3652343734375 0.6679687515625 0.3779296890625 0.6601562484375 0.3798828140625 0.6591796890625 0.3984375015625 0.65625 0.3837890625 0.6494140625 0.388671875 0.6445312484375 0.3818359375 0.6445312484375 0.4326171859375 0.6376953109375 0.4257812484375 0.6289062484375 0.4345703109375 0.6298828140625 0.4492187515625 0.6337890625 0.439453125 0.6396484359375 0.4472656265625 0.6445312484375 0.4443359375 0.6445312484375 0.4912109375 0.6523437515625 0.4931640625 0.6445312484375 0.5107421859375 0.6445312484375 0.5458984359375 0.6523437515625 0.5498046890625 0.6445312484375 0.5517578140625 0.6445312484375 0.5615234359375 0.6523437515625 0.5654296890625 0.6435546890625 0.5742187515625 0.6396484359375 0.5664062484375 0.6298828140625 0.5664062484375 0.6259765640625 0.5898437515625 0.6226337156249999 0.5480581421874999 0.6152343734375 0.5517578140625 0.6210937515625 0.5654296890625 0.6132812484375 0.5673828140625 0.6132812484375 0.5771484359375 0.6210937515625 0.5791015640625 0.6132812484375 0.5830078140625 0.6132812484375 0.5927734359375 0.6289062484375 0.5947265640625 0.6279296890625 0.5996093734375 0.6142578140625 0.5976562484375 0.6132812484375 0.6083984359375 0.6464843734375 0.6123046890625 0.6435546890625 0.6210937515625 0.6396484359375 0.6132812484375 0.6298828140625 0.6132812484375 0.6279296890625 0.6210937515625 0.6162109375 0.6132812484375 0.6142578140625 0.625 0.671875 0.6240234359375 0.6728515640625 0.59375 0.7148437515625 0.5947265640625 0.7060546890625 0.6054687515625 0.7021484359375 0.5976562484375 0.6923828140625 0.5976562484375 0.6923828140625 0.609375
0 0.4833984359375 0.828125 0.4833984359375 0.8164062484375 0.4384765640625 0.8242187515625 0.435546875 0.8134765640625 0.5009765640625 0.810546875 0.5263671859375 0.8125 0.5205078140625 0.828125 0.5615234359375 0.828125 0.5634765640625 0.8125 0.6328124984375 0.8037109375 0.578125 0.7998046890625 0.625 0.7880859375 0.6015624984375 0.7841796890625 0.609375 0.7392578140625 0.5947265640625 0.7402343734375 0.5927734359375 0.7226562484375 0.5712890625 0.7382812484375 0.5703124984375 0.7080078140625 0.5546875015625 0.7060546890625 0.578125 0.6689453109375 0.5634765640625 0.6835937515625 0.5439453109375 0.642578125 0.5205078140625 0.6445312484375 0.5185546890625 0.6679687515625 0.484375 0.6513671859375 0.484375 0.6240234359375 0.5146484359375 0.625 0.5126953109375 0.611328125 0.4697265640625 0.6210937515625 0.4658203109375 0.6035156265625 0.4570312484375 0.6240234359375 0.4794921859375 0.625 0.4716796890625 0.6367187515625 0.4580078140625 0.6289062484375 0.4541015640625 0.6464843734375 0.4521484359375 0.6191406265625 0.4169921859375 0.6601562484375 0.3945312484375 0.6611328140625 0.3935546890625 0.7617187515625 0.3896484359375 0.7226562484375 0.375 0.7451171859375 0.3720703109375 0.705078125 0.3623046890625 0.7265624984375 0.3564453109375 0.705078125 0.34375 0.7236328140625 0.34375 0.5771484359375 0.3476562484375 0.6845703109375 0.3583984359375 0.703125 0.3623046890625 0.607421875 0.3740234359375 0.703125 0.392578125 0.6689453109375 0.3896484359375 0.654296875 0.375 0.6748046890625 0.375 0.6259765640625 0.4453124984375 0.5693359375 0.4296875015625 0.5673828140625 0.4384765640625 0.544921875 0.453125 0.5458984359375 0.4521484359375 0.5351562484375 0.4375 0.5419921859375 0.4384765640625 0.515625 0.4423828140625 0.53125 0.4570312484375 0.5166015640625 0.4609375015625 0.5673828140625 0.4716796890625 0.4765624984375 0.4882812484375 0.4873046890625 0.4970703109375 0.5625 0.5009765640625 0.53125 0.5126953109375 0.5703124984375 0.5615234359375 0.5625 0.5556640625 0.546875 0.5693359375 0.5625 0.5683593734375 0.5478515640625 0.5859375015625 0.5458984359375 0.578125 0.4697265640625 0.5849609375 0.53125 0.59375 0.4052734359375 0.5986328140625 0.421875 0.6240234359375 0.423828125 0.6279296890625 0.3945312484375 0.6289062484375 0.4248046890625 0.640625 0.4189453109375 0.6318359375 0.3808593734375 0.6259765640625 0.392578125 0.6142578140625 0.375 0.6210937515625 0.3935546890625 0.609375 0.3935546890625 0.609375 0.3369140625 0.6240234359375 0.359375 0.6289062484375 0.3291015640625 0.6318359375 0.3691406265625 0.6416015640625 0.3378906265625 0.6445312484375 0.3583984359375 0.65625 0.3525390625 0.65625 0.3017578140625 0.640625 0.3017578140625 0.6337890625 0.2578124984375 0.6445312484375 0.2548828140625 0.6464843734375 0.2939453109375 0.6640624984375 0.2919921859375 0.65625 0.2587890625 0.6318359375 0.25 0.6240234359375 0.2265624984375 0.5341796890625 0.203125 0.3828124984375 0.1943359375 0.4052734359375 0.173828125 0.3388671859375 0.1953124984375 0.3291015640625 0.1835937515625 0.3173828140625 0.2070312484375 0.2900390625 0.205078125 0.2734375015625 0.2216796890625 0.2685546890625 0.2460937515625 0.2578124984375 0.2373046890625 0.2548828140625 0.267578125 0.3085937515625 0.2314453109375 0.2617187515625 0.2880859375 0.2421875015625 0.3564453109375 0.2314453109375 0.3515624984375 0.2128906265625 0.3837890625 0.2060546890625 0.5 0.2001953109375 0.4433593734375 0.1972656265625 0.4580078140625 0.2148437515625 0.6943359375 0.2402343734375 0.7392578140625 0.2617187515625 0.7451171859375 0.2539062484375 0.7548828140625 0.294921875 0.7783203109375 0.2880859375 0.7910156265625 0.2998046890625 0.7851562484375 0.3349609375 0.8203124984375 0.3525390625 0.8222656265625 0.3603515640625 0.796875 0.3642578140625 0.8125 0.3867187515625 0.8134765640625 0.3564453109375 0.8183593734375 0.3652343734375 0.8291015640625 0.390625 0.8271484359375 0.3916015640625 0.796875 0.3955078140625 0.828125 0.4833984359375 0.828125


================================================
FILE: TumorDetection/train/labels/no_tumor_416_jpg.rf.119864a2eece4b1ae1e956cbff256020.txt
================================================
0 0.9874218734375001 0.4150390609375 0.953906253125 0.1962890609375 0.888164059375 0.12304687343750001 0.7180078140624999 0.0390625 0.5942578140625 0.0078125 0.3673828140625 0.011718746875 0.1791796859375 0.07226562656249999 0.025781253124999997 0.2138671875 0.005156253125 0.3193359390625 0.012890626562499998 0.5869140609375 0.11859374687499999 0.7978515609375 0.215273440625 0.8984375 0.3983203140625 0.9824218734375 0.613593746875 0.9873046875 0.7463671859375001 0.941406253125 0.850781253125 0.8603515609375 0.9745312531250001 0.6669921875 0.9874218734375001 0.4150390609375


================================================
FILE: TumorDetection/train/labels/no_tumor_420_jpg.rf.9d4673b08e0deb85104ac03184eda772.txt
================================================
0 0.9462721265624999 0.6083984375 0.9394644156249999 0.5068359359375 0.8396179531250001 0.2021484375 0.7851562484375 0.1103515625 0.6274442312499999 0.029296875 0.40052045625000005 0.021484373437500003 0.24848153125 0.08398437343750001 0.1679235890625 0.1962890640625 0.0658078953125 0.4814453140625 0.0544617046875 0.6201171890625 0.0930387453125 0.7685546859375 0.189481346875 0.8808593734375 0.323366375 0.9472656265625 0.5060400109375001 0.9755859359375 0.715944496875 0.9179687515625 0.82827176875 0.8447265625 0.9213105140625 0.6923828109375 0.9462721265624999 0.6083984375


================================================
FILE: TumorDetection/train/labels/no_tumor_430_jpg.rf.af0e9f050f5cb81fc5df3cd867d32523.txt
================================================
0 0.7480468765625 0.7250976578125 0.7558593765625 0.6752929671875 0.75390625 0.4731445328125 0.72265625 0.2416992171875 0.6318359375 0.08203125 0.5888671875 0.052734375 0.5302734375 0.038085935937499996 0.4287109375 0.052734375 0.3623046875 0.10546875 0.26953125 0.2651367171875 0.23828125 0.4614257828125 0.2363281234375 0.6547851578125 0.24609375 0.6782226578125 0.2480468765625 0.7368164078125 0.2714843765625 0.8012695328125 0.3701171875 0.931640625 0.4404296875 0.966796875 0.4785156234375 0.9682617171875 0.5830078125 0.9580078140625 0.6376953125 0.9228515640625 0.7167968765625 0.7954101578125 0.7480468765625 0.7250976578125


================================================
FILE: TumorDetection/train/labels/no_tumor_457_jpg.rf.96965937cd4c3ab8c2fedc376e998dff.txt
================================================
0 0.9498406421875 0.6376953125 0.9474113828125 0.4365234359375 0.83080690625 0.2021484359375 0.727563353125 0.08984375156249999 0.5769492390625001 0.0273437515625 0.4214766015625 0.021484376562499997 0.251428403125 0.08789062343750001 0.14818485625 0.2138671875 0.0364389 0.4482421875 0.0485851984375 0.6689453125 0.1068874375 0.8115234359375 0.2222772828125 0.9199218765625 0.341311025 0.9550781234375 0.6048857265625001 0.9560546875 0.6789781546875 0.9472656234375 0.7737192937499999 0.9042968765625 0.88910914375 0.7880859359375 0.9498406421875 0.6376953125


================================================
FILE: TumorDetection/train/labels/no_tumor_468_jpg.rf.87ab342161dd8d47ec77853a0f40c082.txt
================================================
0 0.8359375015625 0.5556640625 0.8320312484375 0.2744140625 0.7871093734375 0.1689453109375 0.6826171859375 0.06640624843750001 0.5771484359375 0.021484373437500003 0.4619140625 0.019531248437500003 0.3271484359375 0.07421875156249999 0.216796875 0.1884765640625 0.1914062484375 0.2626953109375 0.1835937515625 0.5498046890625 0.21875 0.6708984359375 0.3115234359375 0.8125 0.4130859375 0.873046875 0.5585937515625 0.8798828140625 0.6572265640625 0.8476562484375 0.7021484359375 0.814453125 0.8007812484375 0.6767578140625 0.8359375015625 0.5556640625


================================================
FILE: TumorDetection/train/labels/no_tumor_473_jpg.rf.ab7cc6c12b14c280edd149b89eb4c0a4.txt
================================================
0 0.845703125 0.3525390625 0.8378906265625 0.2451171859375 0.814453125 0.1787109375 0.6923828140625 0.07031249843750001 0.5810546890625 0.037109373437499996 0.3720703109375 0.048828125 0.2646484359375 0.111328125 0.1953124984375 0.1943359375 0.1503906265625 0.3818359375 0.189453125 0.6650390625 0.2285156265625 0.7314453109375 0.3623046890625 0.8691406265625 0.482421875 0.8935546890625 0.6630859375 0.8710937515625 0.78125 0.7060546890625 0.8320312484375 0.5654296890625 0.845703125 0.3525390625


================================================
FILE: TumorDetection/train/labels/no_tumor_481_jpg.rf.0f3734afa1d75ea772750755d55d62af.txt
================================================
0 0.538673290625 0.880859375 0.6494069859375 0.865234375 0.71630859375 0.8359375 0.7912845328125 0.7783203125 0.8535722390625 0.7080078125 0.8881765187499999 0.6064453125 0.8443444281250001 0.3525390609375 0.7659080609375 0.2119140609375 0.724382928125 0.1591796875 0.6701695546875001 0.119140625 0.5663567140625 0.083984375 0.490227296875 0.0859375 0.46946473125 0.103515625 0.45100911562499996 0.0859375 0.37718665156250003 0.08984374843750001 0.291829428125 0.115234375 0.25491819531250004 0.138671875 0.1522588328125 0.2509765609375 0.0922780796875 0.4580078125 0.083050271875 0.6298828125 0.1384171203125 0.7392578125 0.25953209843749997 0.84375 0.344889321875 0.880859375 0.43140002187500004 0.8896484390625 0.4579299703125 0.884765625 0.494841203125 0.8554687484375 0.5202176734374999 0.880859375 0.538673290625 0.880859375


================================================
FILE: TumorDetection/train/labels/no_tumor_485_jpg.rf.5ae3d38860013bc53cd9c2cdfcade36c.txt
================================================
0 0.8046875015625 0.5244140625 0.7441406265625 0.3662109375 0.7539062484375 0.2939453109375 0.7128906265625 0.1806640625 0.6591796890625 0.1367187515625 0.6376953109375 0.1660156265625 0.5888671859375 0.1367187515625 0.5361328140625 0.1503906265625 0.5205078140625 0.1328124984375 0.4755859375 0.1289062484375 0.3544921859375 0.169921875 0.294921875 0.2373046890625 0.2851562484375 0.3447265640625 0.2265624984375 0.4677734359375 0.2128906265625 0.5830078140625 0.25 0.7197265640625 0.3232421859375 0.8339843734375 0.4267578140625 0.8867187515625 0.4814453109375 0.859375 0.4912109375 0.890625 0.5351562484375 0.9013671859375 0.6142578140625 0.8828124984375 0.7421875015625 0.7744140625 0.798828125 0.6455078140625 0.8046875015625 0.5244140625


================================================
FILE: TumorDetection/train/labels/no_tumor_486_jpg.rf.38c0a3771298af140f9f9478009f49e9.txt
================================================
0 0.8604824437499999 0.4794921890625 0.8083319890625 0.3330078109375 0.8005094234374999 0.2490234359375 0.7457514484375001 0.1826171890625 0.6505768765625 0.130859375 0.40807727812500005 0.119140625 0.3376741703125 0.134765625 0.239892071875 0.2099609375 0.21903189687500002 0.3095703125 0.135591175 0.4658203125 0.1381986953125 0.5908203125 0.2320695046875 0.7685546875 0.3063839015625 0.837890625 0.36896444375000004 0.8671874984375 0.5893000984375 0.8896484359375 0.73792888125 0.8115234359375 0.844837309375 0.6748046875 0.8656974843749999 0.6123046875 0.8604824437499999 0.4794921890625


================================================
FILE: TumorDetection/train/labels/no_tumor_533_jpg.rf.e7545dd4398fa1863e69fc810d45f00d.txt
================================================
0 0.944002890625 0.3662109359375 0.9076950859374999 0.2314453140625 0.8225986703125001 0.126953125 0.666021271875 0.0507812484375 0.46632835312500004 0.0273437515625 0.2825200984375 0.08398437343750001 0.1588466421875 0.1689453140625 0.0816925546875 0.2919921890625 0.0544617046875 0.3798828109375 0.0567309421875 0.4775390640625 0.1429619765625 0.7548828109375 0.2178468203125 0.8955078109375 0.3800973171875 0.9746093734375 0.5990787578125001 0.9814453140625 0.740906115625 0.923828125 0.8418871906250001 0.7861328109375 0.9349259421875 0.5126953140625 0.944002890625 0.3662109359375


================================================
FILE: TumorDetection/train/labels/no_tumor_538_jpg.rf.680e5f1c88286ecbd7433465e9b95b6d.txt
================================================
0 0.7695312484375 0.6845703109375 0.7890624984375 0.6806640625 0.7714843734375 0.6767578140625 0.794921875 0.6708984359375 0.7773437515625 0.6611328140625 0.7861328140625 0.6367187515625 0.8066406265625 0.6357421859375 0.7871093734375 0.6298828140625 0.810546875 0.6259765640625 0.8164062484375 0.5673828140625 0.8339843734375 0.5576171859375 0.814453125 0.5498046890625 0.8359375015625 0.5419921859375 0.8242187515625 0.5400390625 0.8359375015625 0.5263671859375 0.8242187515625 0.5244140625 0.8359375015625 0.5126953109375 0.8242187515625 0.5068359375 0.8359375015625 0.4951171859375 0.8242187515625 0.4912109375 0.830078125 0.3857421859375 0.7929687515625 0.2763671859375 0.6630859375 0.1328124984375 0.6630859375 0.1777343734375 0.5576171859375 0.11523437343750001 0.4951171859375 0.109375 0.4736328140625 0.125 0.4677734359375 0.091796875 0.4794921859375 0.11718750156249999 0.4833984359375 0.08984375156249999 0.4912109375 0.10546875156249999 0.4990234359375 0.08984375156249999 0.5107421859375 0.10156249843750001 0.546875 0.0869140625 0.5126953109375 0.080078125 0.3837890625 0.10156249843750001 0.2734375015625 0.1806640625 0.1953124984375 0.3076171859375 0.1660156265625 0.4287109375 0.1816406265625 0.6083984359375 0.1972656265625 0.6552734359375 0.2226562484375 0.6591796890625 0.21875 0.7021484359375 0.2402343734375 0.7060546890625 0.2421875015625 0.7314453109375 0.3212890625 0.828125 0.4326171859375 0.9023437515625 0.484375 0.9072265640625 0.5908203109375 0.9003906265625 0.6953124984375 0.8388671859375 0.779296875 0.7158203109375 0.7695312484375 0.6845703109375


================================================
FILE: TumorDetection/train/labels/no_tumor_556_jpg.rf.cf153f32249e1cbd3a50e594af847272.txt
================================================
0 0.9417336531249999 0.5107421890625 0.8396179531250001 0.2021484375 0.7715408203125 0.09863281093750001 0.6070210859375 0.021484373437500003 0.3800973171875 0.0273437515625 0.24621229531250002 0.08203124843750001 0.19288520781250001 0.1357421890625 0.0794233171875 0.4150390640625 0.0544617046875 0.6162109359375 0.090769509375 0.7314453140625 0.187212109375 0.859375 0.366481890625 0.9492187515625 0.5446170515625 0.9736328109375 0.743175353125 0.921875 0.8623103359375 0.8330078109375 0.9258489890625 0.7119140640625 0.9417336531249999 0.5107421890625


================================================
FILE: TumorDetection/train/labels/no_tumor_574_jpg.rf.ee8c2c9fb8a2e5f61947a418397ea6f3.txt
================================================
0 0.779296875 0.8486328140625 0.783203125 0.7900390625 0.7939453109375 0.8066406265625 0.8125 0.7744140625 0.8164062484375 0.6708984359375 0.8496093734375 0.5576171859375 0.84375 0.4150390625 0.810546875 0.2900390625 0.7402343734375 0.1513671859375 0.7001953109375 0.11718750156249999 0.6416015640625 0.111328125 0.5927734359375 0.078125 0.3701171859375 0.076171875 0.3486328140625 0.10546875156249999 0.2900390625 0.12109375156249999 0.2861328140625 0.1445312484375 0.2080078140625 0.189453125 0.2099609375 0.21875 0.1982421859375 0.1992187515625 0.171875 0.2041015640625 0.1933593734375 0.2138671859375 0.1796875015625 0.2158203109375 0.1796875015625 0.2490234359375 0.1708984359375 0.236328125 0.158203125 0.2548828140625 0.1679687515625 0.2666015640625 0.1367187515625 0.3994140625 0.140625 0.6474609375 0.15625 0.7314453109375 0.1875 0.7490234359375 0.1513671859375 0.7402343734375 0.140625 0.7626953109375 0.1494140625 0.779296875 0.1767578140625 0.779296875 0.1826171859375 0.7578124984375 0.205078125 0.7822265640625 0.2021484359375 0.8046875015625 0.2392578140625 0.8046875015625 0.2509765640625 0.84375 0.3017578140625 0.861328125 0.3447265640625 0.9140624984375 0.3876953109375 0.923828125 0.4013671859375 0.9453124984375 0.5039062484375 0.9580078140625 0.5302734359375 0.9570312484375 0.5498046890625 0.9316406265625 0.6962890625 0.919921875 0.779296875 0.8486328140625


================================================
FILE: TumorDetection/train/labels/no_tumor_582_jpg.rf.6272f3e2fef0ce02f3da23ae0b9eee8a.txt
================================================
0 0.7558593734375 0.6298828140625 0.8066406265625 0.5224609375 0.814453125 0.4306640625 0.7695312484375 0.2939453109375 0.6630859375 0.1816406265625 0.5869140625 0.1484375015625 0.4443359375 0.138671875 0.3544921859375 0.1679687515625 0.2871093734375 0.2177734359375 0.2246093734375 0.3056640625 0.173828125 0.4189453109375 0.1875 0.4228515640625 0.1757812484375 0.4345703109375 0.2070312484375 0.5732421859375 0.2382812484375 0.6103515640625 0.248046875 0.7197265640625 0.2851562484375 0.7744140625 0.3642578140625 0.8398437515625 0.5234375015625 0.8623046890625 0.6474609375 0.8378906265625 0.71875 0.7724609375 0.7539062484375 0.7119140625 0.7558593734375 0.6298828140625


================================================
FILE: TumorDetection/train/labels/no_tumor_586_jpg.rf.50372a07323a4714564e4b908b5a1116.txt
================================================
0 0.5 0.50390625 1 0.97734375


================================================
FILE: TumorDetection/train/labels/no_tumor_587_jpg.rf.bb4bece0c4ba89ffadc4a9da5d780b7d.txt
================================================
0 0.9862154421874999 0.6025390625 0.9703087453125001 0.3681640625 0.88850285625 0.2158203140625 0.8021521921874999 0.1240234375 0.7055757984375 0.0566406234375 0.5328744765625 0.009765623437499999 0.3329045234375 0.0273437515625 0.21928523125 0.0859375 0.12043644687499999 0.1865234375 0.0159067015625 0.4033203140625 0.0181790859375 0.6474609375 0.0658991875 0.7724609375 0.12043644687499999 0.8564453140625 0.255643403125 0.9453125 0.38516939375 0.984375 0.46129432187499997 0.9892578140625 0.6510385390625 0.9785156234375 0.8510084906250001 0.8984375 0.94076773125 0.7783203140625 0.9862154421874999 0.6025390625


================================================
FILE: TumorDetection/train/labels/no_tumor_58_jpg.rf.65cbd9ab986d64b73ec16aa4de1e42b3.txt
================================================
0 0.8774671046875 0.6787109375 0.9023437515625 0.5888671859375 0.8955592109375001 0.4619140625 0.859375 0.3740234375 0.82771381875 0.3564453140625 0.8164062484375 0.2919921859375 0.773437503125 0.2607421859375 0.7689144734375 0.2138671859375 0.7304687515625 0.1650390625 0.64113898125 0.10351562343750001 0.5574629953125 0.08789062343750001 0.5190172671875 0.1015625 0.491879109375 0.1347656234375 0.43986431093749995 0.09375 0.338096215625 0.11328124843750001 0.289473684375 0.1396484375 0.2600740140625 0.1904296859375 0.22841282812500002 0.2021484375 0.1741365125 0.2958984375 0.0972450671875 0.5107421859375 0.1221217125 0.6474609375 0.24198190937500003 0.8037109375 0.39237253125 0.8945312484375 0.4715254921875 0.9023437515625 0.45682565625000005 0.9169921859375 0.4805715453125 0.9277343765625 0.5518092109375 0.9326171859375 0.6275699 0.9199218765625 0.629831415625 0.9023437515625 0.7643914453125 0.8291015625 0.8774671046875 0.6787109375


================================================
FILE: TumorDetection/train/labels/no_tumor_591_jpg.rf.3a2c2b34baf0239f388124c41cfb62f6.txt
================================================
0 0.9109724828125 0.3681640640625 0.8696750640625 0.2470703125 0.7445681734375 0.11914062343750001 0.5380810796875 0.0605468765625 0.4919251390625 0.1289062484375 0.45548623906249996 0.06640624843750001 0.402042521875 0.0605468765625 0.2441406265625 0.1015625 0.1287507765625 0.2119140640625 0.07044853749999999 0.3564453125 0.068019278125 0.5283203125 0.1433263390625 0.6435546875 0.1749067171875 0.7666015640625 0.23320895468749997 0.8486328125 0.3996132640625 0.9199218765625 0.4481984625 0.9199218765625 0.4822081 0.890625 0.4992129203125 0.9160156234375 0.544154228125 0.9208984359375 0.649827034375 0.9042968765625 0.774933925 0.8173828125 0.9109724828125 0.5439453125 0.9109724828125 0.3681640640625


================================================
FILE: TumorDetection/train/labels/no_tumor_593_jpg.rf.4dd8f671d519a783966a473992fd0ff7.txt
================================================
0 0.8339843765625 0.6095340765625 0.8613281234375 0.3314656375 0.81640625 0.2025063625 0.7275390609375 0.09268947968749999 0.6474609390625 0.044329751562500005 0.4951171890625 0.012089932812499999 0.3701171890625 0.044329751562500005 0.3095703109375 0.08462952500000001 0.26171875 0.1440716921875 0.21484375 0.2428061359375 0.1953125 0.4785598140625 0.2480468765625 0.7485682984375 0.2890625 0.8311828328124999 0.4189453109375 0.9228648187499999 0.54296875 0.93596224375 0.6142578109375 0.926894796875 0.6787109390625 0.894654978125 0.75390625 0.8110329453125 0.8339843765625 0.6095340765625


================================================
FILE: TumorDetection/train/labels/no_tumor_601_jpg.rf.daafa5bad1f585f666ca475a583d1649.txt
================================================
0 0.9065888062499999 0.8154296890625 0.9552475312500001 0.7138671875 0.9706134390625 0.5068359375 0.9193937343749999 0.3076171875 0.8271982609374999 0.1396484359375 0.6722586484375 0.041015626562500004 0.57238021875 0.021484373437500003 0.43152603125000005 0.021484373437500003 0.2650619796875 0.06640624843750001 0.179268975 0.1259765640625 0.13573222499999998 0.1806640625 0.05378069375 0.3544921875 0.0358537953125 0.6064453109375 0.0742685765625 0.7705078125 0.1831104484375 0.888671875 0.341891540625 0.9648437515625 0.5736607140625 0.9833984359375 0.6568927390625 0.9746093734375 0.7414052546875001 0.9414062484375 0.861771565625 0.8652343734375 0.9065888062499999 0.8154296890625


================================================
FILE: TumorDetection/train/labels/no_tumor_605_jpg.rf.e722600b49a9c1465deaaf5c11599292.txt
================================================
0 0.88310455625 0.4013671859375 0.7861784484375 0.2431640640625 0.69248320625 0.1640625 0.6084805796875 0.13671875 0.5007849015625 0.14453125 0.3844735703125 0.1269531265625 0.291855284375 0.1640625 0.20031396250000003 0.2451171859375 0.1292348125 0.3603515640625 0.11415742031250001 0.4423828140625 0.13138872968749998 0.6025390640625 0.198160046875 0.7724609359375 0.3198561640625 0.8828125 0.440475321875 0.91796875 0.49647707499999993 0.90625 0.5729410046875001 0.9169921859375 0.6774058125 0.8964843734375 0.7926401875 0.8017578140625 0.8852584671875 0.5517578140625 0.88310455625 0.4013671859375


================================================
FILE: TumorDetection/train/labels/no_tumor_610_jpg.rf.1ad423b5823a71bf739a909d7065f5cd.txt
================================================
0 0.83203125 0.5084605437499999 0.79296875 0.36646745625 0.6767578109375 0.1545218890625 0.5419921890625 0.0814372125 0.3994140609375 0.0814372125 0.3056640609375 0.123199884375 0.2421875 0.184799825 0.134765625 0.4270233265625 0.11328125 0.57736895 0.2099609390625 0.8770161296874999 0.3466796890625 0.9563652078124999 0.517578125 0.9720262125 0.6279296890625 0.941748271875 0.7578125 0.8404737875 0.828125 0.602426553125 0.83203125 0.5084605437499999


================================================
FILE: TumorDetection/train/labels/no_tumor_61_jpg.rf.2b8a81a6ad68256a0cc0204cbc55df6a.txt
================================================
0 0.8242187484375 0.5359860984375 0.7851562515625 0.3299176578125 0.6953125015625 0.1259307140625 0.5693359390625 0.037466990625 0.4287109390625 0.033303990625 0.3427734359375 0.062444981249999996 0.2539062515625 0.1592347046875 0.162109375 0.3777921453125 0.1425781234375 0.6296535734375001 0.1855468765625 0.794092028125 0.2880859390625 0.9325117374999999 0.3837890609375 0.9824677234375001 0.515625 0.993915971875 0.6201171890625 0.9595712296875 0.7304687484375 0.879433503125 0.8164062515625 0.6733650609375 0.8242187484375 0.5359860984375


================================================
FILE: TumorDetection/train/labels/no_tumor_625_jpg.rf.aed2835c83bf66f675373c82b57c5c31.txt
================================================
0 0.5830078125 0.864002546875 0.6142578125 0.864002546875 0.6904296875 0.8239096218749999 0.73046875 0.7888283125 0.77734375 0.726684278125 0.81640625 0.6344705484375 0.81640625 0.5061731875000001 0.78125 0.41195481093749997 0.7441406265625 0.3598340078125 0.7363281265625 0.3237503734375 0.7402343734375 0.269624925 0.7207031265625 0.20948553749999999 0.6630859375 0.154357765625 0.5712890625 0.122283425 0.5146484375 0.1182741296875 0.4755859375 0.13832059531250002 0.4501953125 0.1262927140625 0.3974609375 0.1343113 0.3681640625 0.160371703125 0.3486328125 0.1503484703125 0.3232421875 0.156362409375 0.2900390625 0.18843675 0.2402343734375 0.25158310781250004 0.1953125 0.454052384375 0.1796875 0.47409884687500004 0.16796875 0.5282242953125 0.171875 0.6725588265625 0.2109375 0.766777203125 0.2998046875 0.8419514390625 0.3837890625 0.8760304265625001 0.45703125 0.8890606265625 0.4677734375 0.8820443640625 0.5556640625 0.8800397171875 0.5771484375 0.8559839609375001 0.5830078125 0.864002546875
4 0.5390625 0.40794551874999996 0.54296875 0.3678525921875 0.5800781265625 0.3277596671875 0.58203125 0.305708559375 0.5693359375 0.3006969421875 0.5576171875 0.30871552812499997 0.4990234375 0.374868853125 0.4013671875 0.31072017343749997 0.3964843734375 0.345801484375 0.4355468734375 0.371861884375 0.44921875 0.41195481093749997 0.421875 0.5182010640625 0.3925781265625 0.5743311609374999 0.39453125 0.5923729765625 0.4267578125 0.5893660078125 0.5048828125 0.5151940953125 0.5234375 0.5562893421874999 0.5732421875 0.605403178125 0.5957031265625 0.6084101468750001 0.6015625 0.59036833125 0.56640625 0.534238234375 0.55078125 0.4901360171875 0.5390625 0.40794551874999996


================================================
FILE: TumorDetection/train/labels/no_tumor_634_jpg.rf.e499d2363eec7206db37a25261614a96.txt
================================================
0 0.9414451187499999 0.5654296890625 0.94388409375 0.3623046890625 0.882909671875 0.2119140640625 0.8304716671874999 0.1484374984375 0.6353535078125 0.0507812515625 0.42072353281249997 0.035156251562500004 0.2475561640625 0.078125 0.12926578125 0.1630859359375 0.0512185171875 0.3232421875 0.0341456765625 0.5166015625 0.0731693078125 0.6806640640625 0.1902402046875 0.8759765625 0.34755421875 0.9511718765625 0.5438918703125 0.9677734375 0.6451094125 0.9511718765625 0.7329125875 0.912109375 0.836569109375 0.8095703109375 0.9414451187499999 0.5654296890625


================================================
FILE: TumorDetection/train/labels/no_tumor_638_jpg.rf.db54847287e9389b62df0d9f36400b01.txt
================================================
0 0.954699159375 0.3955078125 0.88910914375 0.2099609359375 0.749426696875 0.0859375 0.610958875 0.041015623437499996 0.3461695421875 0.042968751562500004 0.22470654687499997 0.08398437656249999 0.11417521875 0.1826171875 0.053443721875 0.3232421875 0.02915111875 0.5146484359375 0.0728778015625 0.6494140640625 0.216204134375 0.8857421875 0.3510280640625 0.9609375 0.507715328125 0.9814453125 0.657114815625 0.9492187515625 0.7943680046875 0.8466796875 0.9425528609375 0.5712890640625 0.954699159375 0.3955078125


================================================
FILE: TumorDetection/train/labels/no_tumor_640_jpg.rf.778d020be48fc5d6f89b1fdaa06531a4.txt
================================================
0 0.8852584671875 0.4033203140625 0.77971670625 0.2353515640625 0.6774058125 0.1542968734375 0.597711009375 0.1347656265625 0.49432315937500004 0.1425781265625 0.388781396875 0.125 0.2875474578125 0.1640625 0.198160046875 0.2451171859375 0.14646612187500002 0.3251953140625 0.11415742031250001 0.3955078140625 0.1292348125 0.6005859359375 0.2089296171875 0.7880859359375 0.304778765625 0.87890625 0.43832140624999993 0.91796875 0.5729410046875001 0.9169921859375 0.6774058125 0.8964843734375 0.7904862703125 0.8017578140625 0.8852584671875 0.5634765640625 0.8852584671875 0.4033203140625


================================================
FILE: TumorDetection/train/labels/no_tumor_689_jpg.rf.b93d4818eca1c52a004e876d2ff69881.txt
================================================
0 0.8046875015625 0.5263671859375 0.7441406265625 0.3662109375 0.7539062484375 0.2998046890625 0.7148437515625 0.1826171859375 0.6728515640625 0.1464843734375 0.6396484359375 0.1679687515625 0.5888671859375 0.1367187515625 0.5361328140625 0.1523437515625 0.5205078140625 0.1328124984375 0.4755859375 0.1289062484375 0.3544921859375 0.171875 0.298828125 0.2294921859375 0.283203125 0.3505859375 0.216796875 0.5029296890625 0.2148437515625 0.5966796890625 0.2753906265625 0.7705078140625 0.3408203109375 0.8476562484375 0.4267578140625 0.8867187515625 0.4833984359375 0.8554687515625 0.5029296890625 0.8964843734375 0.5742187515625 0.8994140625 0.6787109375 0.8378906265625 0.751953125 0.7568359375 0.8007812484375 0.6357421859375 0.8046875015625 0.5263671859375


================================================
FILE: TumorDetection/train/labels/no_tumor_698_jpg.rf.3c0efb1fb3525c0ef805bb505130b347.txt
================================================
0 0.893047625 0.8623046859375 0.9816706734375 0.6533203140625 0.9793982890624999 0.4169921859375 0.9612192015625001 0.3427734375 0.8907752406249999 0.2197265625 0.7760197546875001 0.10351562343750001 0.7055757984375 0.0566406234375 0.605590821875 0.019531248437500003 0.41016563906250003 0.0136718765625 0.330632134375 0.0292968765625 0.242009090625 0.0703125 0.12043644687499999 0.1884765625 0.0159067015625 0.4072265625 0.0159067015625 0.6298828140625 0.07044396093750001 0.7880859375 0.1624755859375 0.8964843765625 0.251098634375 0.9492187515625 0.360173153125 0.9824218765625 0.5635516828125 0.9892578140625 0.671490009375 0.9765625 0.7714749859375 0.9414062484375 0.8510084906250001 0.9042968765625 0.893047625 0.8623046859375


================================================
FILE: TumorDetection/train/labels/no_tumor_712_jpg.rf.7c9beb139eaeef812a1e03655d4ebe6a.txt
================================================
0 0.8496093734375 0.8349351781250001 0.88671875 0.752744678125 0.9082031265625 0.6404844875 0.91015625 0.55228005 0.89453125 0.46407561406250003 0.86328125 0.3638433 0.81640625 0.2555924 0.7871093734375 0.227527353125 0.78515625 0.1934483671875 0.74609375 0.12929968593750002 0.6689453125 0.058134743749999995 0.6123046875 0.0280650484375 0.5732421875 0.016037170312499998 0.4580078125 0.010023232812500001 0.4052734375 0.0200464625 0.3466796875 0.0461068640625 0.2958984375 0.0801858515625 0.26171875 0.11727180781249999 0.1503906265625 0.3478061296875 0.11132812656249999 0.5101824796875001 0.109375 0.566312575 0.15234375 0.766777203125 0.19140625 0.8289212375 0.2451171875 0.8900629500000001 0.3173828125 0.942183753125 0.3857421875 0.9742580937499999 0.4541015625 0.992299909375 0.5292968734375 0.9933022328125001 0.6162109375 0.9802720328125 0.7001953125 0.95421163125 0.7783203125 0.9101094125 0.8496093734375 0.8349351781250001


================================================
FILE: TumorDetection/train/labels/no_tumor_716_jpg.rf.68ab707e5d1310f0a3edb3cb5c1b8bb0.txt
================================================
0 0.8515624984375 0.4130859375 0.794921875 0.2607421859375 0.6923828140625 0.1367187515625 0.5712890625 0.08398437343750001 0.4150390625 0.08203124843750001 0.2939453109375 0.1367187515625 0.189453125 0.2568359375 0.1523437515625 0.3466796890625 0.1445312484375 0.5498046890625 0.173828125 0.7158203109375 0.2314453109375 0.8125 0.3955078140625 0.9101562484375 0.5273437515625 0.9169921859375 0.6435546890625 0.8964843734375 0.779296875 0.7919921859375 0.8359375015625 0.6669921859375 0.8515624984375 0.4130859375


================================================
FILE: TumorDetection/train/labels/no_tumor_720_jpg.rf.0ae4144b6b9d6f92f062e0572363fbb6.txt
================================================
0 0.8760937500000001 0.8310546859375 0.9745312500000001 0.6396484359375 0.989296875 0.3056640640625 0.947460940625 0.2001953140625 0.7788867203125001 0.05078125 0.6459960953125 0.0058593734375 0.3851367203125 0.0058593734375 0.3113085953125 0.01953125 0.2030273453125 0.06835937343750001 0.054140625 0.2001953140625 0.012304690625 0.3076171859375 0.014765625 0.5166015640625 0.06398437500000001 0.7021484359375 0.191953125 0.8857421859375 0.2620898453125 0.9355468734375 0.3679101546875 0.98046875 0.5709375 0.9951171859375 0.6681445296875 0.98046875 0.7493554703124999 0.9433593734375 0.826875 0.8916015640625 0.8760937500000001 0.8310546859375


================================================
FILE: TumorDetection/train/labels/no_tumor_731_jpg.rf.9edfa6052c902c46fd00b63f823cfa52.txt
================================================
0 0.791015625 0.7119140625 0.82421875 0.6552734375 0.806640625 0.5966796875 0.8203125 0.5927734375 0.81640625 0.4658203125 0.7421875 0.2001953125 0.6923828125 0.134765625 0.6044921875 0.091796875 0.3525390625 0.087890625 0.259765625 0.1787109375 0.255859375 0.2392578125 0.220703125 0.2626953125 0.201171875 0.4970703125 0.21875 0.7880859375 0.23828125 0.7861328125 0.2392578125 0.7578125 0.2578125 0.8251953125 0.3466796875 0.912109375 0.564453125 0.9150390625 0.6650390625 0.90625 0.75 0.7939453125 0.78125 0.7841796875 0.791015625 0.7119140625


================================================
FILE: TumorDetection/train/labels/no_tumor_732_jpg.rf.3c474029a8875f7f6d2e0f57a1cd7c55.txt
================================================
0 0.9184801531250001 0.6767578125 0.9453505828125 0.5419921875 0.9380222843749999 0.4306640625 0.9038235546875001 0.3271484359375 0.85741099375 0.2294921875 0.7987846015625 0.1435546875 0.7511506578125 0.10156249843750001 0.6998525625 0.07421875 0.6363406375 0.05078125 0.54107275 0.03515625 0.416491665625 0.039062498437499996 0.326109309375 0.05859375 0.22595588906250003 0.10156249843750001 0.1807647109375 0.1376953125 0.068397459375 0.3388671875 0.0488553265625 0.3994140625 0.0390842625 0.4892578125 0.0537408609375 0.6611328125 0.14900875 0.8388671875 0.2283986546875 0.8984375015625 0.3114527109375 0.935546875 0.40672059843749997 0.95703125 0.49832433906249995 0.9599609375 0.6045846734375 0.951171875 0.672982134375 0.9277343734375 0.828097796875 0.8388671875 0.9184801531250001 0.6767578125


================================================
FILE: TumorDetection/train/labels/no_tumor_738_jpg.rf.0b4c4930b7a5b70b20147eaa1f5a7a11.txt
================================================
0 0.947564571875 0.6435546890625 0.96549146875 0.3837890625 0.9296376734375 0.2275390625 0.8886619078125 0.1728515640625 0.7875029890625 0.095703125 0.608234015625 0.0273437515625 0.380306321875 0.021484373437500003 0.24201311250000002 0.07031249843750001 0.09219546875 0.1884765640625 0.046097734375 0.2919921875 0.030731826562499998 0.4775390625 0.09219546875 0.7255859375 0.174147 0.8642578125 0.3188426734375 0.9609375015625 0.39823321875 0.982421875 0.5378069187500001 0.9892578125 0.7209173718749999 0.9453124984375 0.8118323515625001 0.8837890625 0.87841796875 0.8017578125 0.947564571875 0.6435546890625


================================================
FILE: TumorDetection/train/labels/no_tumor_749_jpg.rf.d4602efed782d5dabb2361c2a4de72d6.txt
================================================
0 0.9600652328124999 0.6162109390625 0.9670054609375001 0.4462890609375 0.9161104375000001 0.2666015609375 0.7854027625 0.1015625 0.6928663546875 0.0566406234375 0.58876289375 0.03125 0.5771958453125 0.0058593765625 0.44764487343750003 0.0078125 0.4337644125 0.0292968765625 0.42451077187499997 0.009765623437499999 0.3990632578125 0.0058593765625 0.39674985 0.0292968765625 0.2695122859375 0.0703125 0.19779657343749998 0.12304687656249999 0.097163228125 0.2451171890625 0.048581610937500005 0.3779296890625 0.0347011546875 0.5380859390625 0.0208206890625 0.5478515609375 0.0300743296875 0.6513671890625 0.0925364078125 0.7529296890625 0.2082069171875 0.8701171890625 0.20010998125 0.9042968765625 0.23712454374999997 0.9140625 0.267198878125 0.8984375 0.38980961718749996 0.9453125 0.4545851015625 0.9453125 0.47309238281250005 0.96875 0.5054801265625 0.9609375 0.504323421875 0.9755859390625 0.688239534375 0.9394531234375 0.7854027625 0.8828125 0.8744690531250001 0.8037109390625 0.9600652328124999 0.6162109390625


================================================
FILE: TumorDetection/train/labels/no_tumor_74_jpg.rf.9a21362dc700814773b0d227d15ac891.txt
================================================
0 0.9589843734375 0.5217556437499999 0.94921875 0.39158347656250003 0.8574218734375 0.1845883984375 0.7451171859375 0.064019096875 0.6630859359375 0.014937787499999999 0.4638671859375 0.0277416109375 0.2099609359375 0.14724392656250002 0.11914062656249999 0.233669703125 0.0527343734375 0.3681098109375 0.048828126562500004 0.5110857921874999 0.09179687343750001 0.666865596875 0.0234375 0.8397171578125 0.0234375 0.8802625875000001 0.049804685937499996 0.904803240625 0.1533203140625 0.938946759375 0.1728515640625 0.9901620375 0.74609375 0.9954969625000001 0.8652343734375 0.98269314375 0.859375 0.8717267078125 0.87890625 0.7138129359375001 0.9589843734375 0.5217556437499999


================================================
FILE: TumorDetection/train/labels/no_tumor_762_jpg.rf.8c71bc2d60f76136bec76600a41d3877.txt
================================================
0 0.8785546906249999 0.6083984359375 0.9253125000000001 0.4814453140625 0.9326953093749999 0.4013671859375 0.9179296906250001 0.2900390640625 0.79734375 0.1455078140625 0.6583007796875 0.0546875 0.5303320296875 0.0546875 0.5401757796875 0.037109373437499996 0.4983398453125 0.033203126562500004 0.44419922031250003 0.0390625 0.44173827968749996 0.0546875 0.3482226546875 0.0566406265625 0.3137695296875 0.08789062656249999 0.20794922031250002 0.1328125 0.0984375 0.2646484359375 0.0590625 0.3818359359375 0.103359375 0.6376953140625 0.155039059375 0.7041015640625 0.209179690625 0.8251953140625 0.3605273453125 0.9199218734375 0.43927734531249996 0.9316406265625 0.5155664046874999 0.921875 0.529101559375 0.9345703140625 0.7124414046875 0.89453125 0.794882809375 0.8388671859375 0.8711718749999999 0.7099609359375 0.8785546906249999 0.6083984359375


================================================
FILE: TumorDetection/train/labels/no_tumor_765_jpg.rf.350e4653aca3276394180480ff9af8e1.txt
================================================
0 0.8398437515625 0.4853515640625 0.8359375015625 0.3681640625 0.810546875 0.2197265640625 0.7216796890625 0.11914062656249999 0.6181640625 0.06835937343750001 0.4755859375 0.0566406265625 0.3564453109375 0.08203124843750001 0.2089843734375 0.2041015640625 0.1660156265625 0.3896484359375 0.1660156265625 0.5810546890625 0.2304687515625 0.7548828140625 0.3095703109375 0.8554687515625 0.4189453109375 0.9257812484375 0.5410156265625 0.9345703109375 0.6806640625 0.8710937515625 0.763671875 0.7685546890625 0.8183593734375 0.6240234359375 0.8398437515625 0.4853515640625


================================================
FILE: TumorDetection/train/labels/no_tumor_776_jpg.rf.d61b6a7cbab2784da79110fcf613ff50.txt
================================================
0 0.7262500000000001 0.7548828125 0.7451136359375 0.7275390625 0.7710511359375001 0.6845703125 0.7757670453125 0.6494140625 0.7969886359375 0.6220703125 0.8323579546875 0.5263671875 0.84650568125 0.4736328125 0.84650568125 0.4326171875 0.8535795453125001 0.4248046875 0.851221590625 0.3896484375 0.82056818125 0.2958984375 0.778125 0.2294921875 0.7392187484375 0.1953125 0.6944176140625 0.166015625 0.6213210218749999 0.1328125 0.5647301125 0.1328125 0.5340767046875 0.15625 0.49163352343750005 0.12890625 0.4609801125 0.126953125 0.41146306875 0.14453125 0.35015625156249996 0.1796875 0.29474431874999996 0.2275390625 0.25465909062500003 0.2900390625 0.21457386406249998 0.4111328125 0.22400568125000003 0.5478515625 0.25465909062500003 0.6201171875 0.2735227265625 0.7099609375 0.29946022656250004 0.7568359375 0.34308238593749996 0.79296875 0.38552556718750003 0.810546875 0.42561079687500003 0.806640625 0.47984374843750005 0.810546875 0.5057812515625 0.798828125 0.5458664765625 0.8203125 0.5706249999999999 0.8193359375 0.6213210218749999 0.8125 0.6425426140625 0.798828125 0.6755539781250001 0.7890625 0.7262500000000001 0.7548828125


================================================
FILE: TumorDetection/train/labels/no_tumor_779_jpg.rf.43be10e011c63f87810a9c2bef9a85ec.txt
================================================
0 0.883717390625 0.7841796890625 0.9085525359375 0.7724609390625 0.9189005140625 0.6748046890625 0.94166606875 0.6767578140625 0.943735665625 0.4228515625 0.9189005140625 0.4169921859375 0.9023437515625 0.3447265625 0.8526734499999999 0.2451171859375 0.8050727484375001 0.2041015625 0.80300315625 0.1650390609375 0.741950078125 0.1328125015625 0.7295325046875 0.10546874843750001 0.636400696875 0.07226562343750001 0.4335803125 0.07031250156249999 0.379770821875 0.107421875 0.3011261828125 0.111328125 0.2586994671875 0.1455078140625 0.256629871875 0.1708984375 0.1552196828125 0.2529296890625 0.14901089375 0.2939453109375 0.11796695937499999 0.3251953109375 0.11796695937499999 0.3974609390625 0.0848534265625 0.4248046890625 0.0848534265625 0.7177734375 0.11796695937499999 0.8173828140625 0.14694129843749998 0.8408203109375 0.151080490625 0.8876953109375 0.236968715625 0.953125 0.29491739375 0.966796875 0.3073349671875 1 0.4304759171875 0.9990234375 0.617774334375 0.984375 0.7647156328125 0.9296874984375 0.860951834375 0.8505859390625 0.883717390625 0.7841796890625


================================================
FILE: TumorDetection/train/labels/no_tumor_785_jpg.rf.d6e18a06d2588adc74b2b0d25ba1a284.txt
================================================
0 0.8574218734375 0.8289212375 0.8925781265625 0.7547493265624999 0.91015625 0.692605290625 0.9199218734375 0.6064055015625 0.9121093734375 0.3678525921875 0.8740234375 0.322748053125 0.84375 0.307713203125 0.80078125 0.22953199999999999 0.7792968734375 0.153355440625 0.6943359375 0.010023232812500001 0.5732421875 0.002004646875 0.2705078125 0.0060139375 0.25390625 0.0210487859375 0.25390625 0.0912114046875 0.23828125 0.11727180781249999 0.1582031265625 0.1734019046875 0.1484375 0.3097178515625 0.10351562656249999 0.520205709375 0.11914062656249999 0.6545170125 0.1328125 0.6765681203125 0.15234375 0.7747957890625 0.2128906265625 0.8569862859375 0.2626953125 0.9061001203125001 0.3115234375 0.9401791078124999 0.4033203125 0.982276678125 0.4404296875 0.992299909375 0.5332031265625 0.9933022328125001 0.6123046875 0.9842813234375001 0.7060546875 0.95421163125 0.7861328125 0.9061001203125001 0.8574218734375 0.8289212375


================================================
FILE: TumorDetection/train/labels/no_tumor_78_jpg.rf.fd979dc240879db328f9abd46febcda1.txt
================================================
0 0.9980468765625 0.5577935093749999 0.994140625 0.37504973124999996 0.9570312515625 0.2794279859375 0.9023437484375 0.2008056625 0.8115234390625 0.110496240625 0.6787109359375 0.03612376875 0.6728515609375 0.0127495640625 0.4365234390625 0.0063747828125 0.3115234390625 0.0297489859375 0.2119140640625 0.0786223234375 0.10742187656249999 0.16680681875 0.0546874984375 0.28367784218750003 0.0605468765625 0.3941740828125 0.087890625 0.44092249062500005 0.1708984390625 0.5142324953124999 0.2451171875 0.5142324953124999 0.3310546890625 0.478108721875 0.365234375 0.5429190187499999 0.3681640640625 0.58435510625 0.4248046890625 0.54398148125 0.443359375 0.5599184390625 0.5273437484375 0.989153828125 0.583984375 0.99340368125 0.7861328125 0.8244719359375001 0.8779296890625 0.7224754078125 0.9550781234375 0.6789143890625 0.9980468765625 0.5577935093749999


================================================
FILE: TumorDetection/train/labels/no_tumor_802_jpg.rf.07031c4a1ce5d9c87a5b5ebd754a9373.txt
================================================
0 0.8007812484375 0.7509765640625 0.841796875 0.6259765640625 0.84375 0.4970703125 0.8242187515625 0.3798828125 0.798828125 0.2919921875 0.7695312484375 0.2294921875 0.6923828125 0.1523437515625 0.6201171875 0.11328124843750001 0.5615234359375 0.095703125 0.4345703125 0.09765624843750001 0.3369140640625 0.1445312484375 0.2880859359375 0.181640625 0.244140625 0.2294921875 0.2148437515625 0.2802734359375 0.173828125 0.3876953125 0.1601562484375 0.4794921875 0.1601562484375 0.5791015640625 0.1835937515625 0.7197265640625 0.2226562484375 0.8017578125 0.2783203125 0.849609375 0.3662109359375 0.8984375 0.4287109359375 0.919921875 0.4921875 0.9248046875 0.5947265640625 0.9101562484375 0.6513671875 0.8867187515625 0.7099609359375 0.8476562484375 0.8007812484375 0.7509765640625


================================================
FILE: TumorDetection/train/labels/no_tumor_805_jpg.rf.07d98ea520cc63e0d83d9fb307511dc0.txt
================================================
0 0.6581858406250001 0.8603515625 0.6483914093750001 0.8349609375 0.7385001828125 0.7099609375 0.760047934375 0.6376953125 0.760047934375 0.5576171875 0.7110757765625 0.4423828125 0.6914869078125 0.3017578125 0.6542680671875 0.2451171875 0.5827687140625 0.1953125 0.52596100625 0.1816406234375 0.49070105 0.1972656234375 0.45739998281250005 0.1777343765625 0.43977000312500003 0.1796875 0.3545584421875 0.2197265625 0.2918740765625 0.3095703125 0.28207964531249996 0.3798828125 0.26053189374999997 0.4267578125 0.2272308265625 0.4326171875 0.258573009375 0.4423828125 0.24290191718749998 0.4658203125 0.24877857812499998 0.4912109375 0.23310748750000002 0.5380859375 0.221354165625 0.5439453125 0.2311485984375 0.5478515625 0.2193952796875 0.5595703125 0.22918971093749999 0.5654296875 0.23506637187499999 0.6611328125 0.221354165625 0.6669921875 0.23898414531249998 0.6748046875 0.2801207609375 0.7607421875 0.39471561718750003 0.8613281234375 0.4103867078125 0.8886718765625 0.47502995937500003 0.9042968765625 0.57493316875 0.89453125 0.5896248140625 0.9033203125 0.60921368125 0.8994140625 0.6062753515625 0.875 0.61606978125 0.8847656234375 0.6581858406250001 0.8603515625


================================================
FILE: TumorDetection/train/labels/no_tumor_812_jpg.rf.895d53fe327f2d4123f787e9282daeb0.txt
================================================
0 0.716796875 0.8369140625 0.755859375 0.7763671875 0.818359375 0.6259765625 0.8476562515625 0.5048828125 0.849609375 0.4404296875 0.814453125 0.2783203125 0.78125 0.2177734375 0.7138671875 0.1445312515625 0.6669921875 0.11328125156249999 0.5869140625 0.080078125 0.4287109375 0.076171875 0.3818359375 0.0859375 0.3193359375 0.1171875 0.2841796875 0.1445312515625 0.21875 0.2177734375 0.1796875 0.2998046875 0.1601562515625 0.3955078125 0.15625 0.5048828125 0.171875 0.5849609375 0.203125 0.6591796875 0.2109375 0.6962890625 0.234375 0.7568359375 0.2734375 0.8212890625 0.3173828125 0.8671875 0.3642578125 0.8945312515625 0.4345703125 0.916015625 0.4921875 0.9208984375 0.5576171875 0.9179687484375 0.6220703125 0.9023437484375 0.6787109375 0.873046875 0.716796875 0.8369140625


================================================
FILE: TumorDetection/train/labels/no_tumor_814_jpg.rf.c75d6c6bb5fa8e38659cedf59e3397b3.txt
================================================
0 0.8085937515625 0.7099609375 0.8183593734375 0.5810546890625 0.841796875 0.5537109375 0.845703125 0.3095703109375 0.8203124984375 0.2841796890625 0.8085937515625 0.1630859375 0.7353515640625 0.08203124843750001 0.6806640625 0.06835937343750001 0.6552734359375 0.039062498437499996 0.3408203109375 0.048828125 0.21875 0.1455078140625 0.1816406265625 0.2177734359375 0.1777343734375 0.2861328140625 0.1484375015625 0.2978515640625 0.1484375015625 0.6064453109375 0.1796875015625 0.6396484359375 0.1875 0.7255859375 0.21875 0.7548828140625 0.220703125 0.8037109375 0.248046875 0.8154296890625 0.2919921859375 0.8847656265625 0.3935546890625 0.9296875015625 0.466796875 0.9306640625 0.6064453109375 0.9179687515625 0.7001953109375 0.8554687515625 0.7070312484375 0.8662109375 0.8085937515625 0.7099609375


================================================
FILE: TumorDetection/train/labels/no_tumor_828_jpg.rf.0bd5c0221ab1fcd8f6bcd598e21b57fe.txt
================================================
0 0.7226562484375 0.3681640625 0.6572265640625 0.2304687515625 0.5888671859375 0.189453125 0.5283203109375 0.1796875015625 0.4990234359375 0.1972656265625 0.4423828140625 0.173828125 0.3544921859375 0.2089843734375 0.2890624984375 0.2939453109375 0.2460937515625 0.4873046890625 0.25 0.6298828140625 0.2890624984375 0.7255859375 0.3388671859375 0.7890624984375 0.4228515640625 0.8339843734375 0.4833984359375 0.8183593734375 0.544921875 0.8349609375 0.6455078140625 0.7871093734375 0.71875 0.6884765640625 0.7421875015625 0.5673828140625 0.7226562484375 0.3681640625


================================================
FILE: TumorDetection/train/labels/no_tumor_833_jpg.rf.b51277ee33ce606516cd652c21a0dc1c.txt
================================================
0 0.73828125 0.6399972078125 0.72265625 0.381905690625 0.6640625 0.1342773453125 0.6259765640625 0.0627790171875 0.5595703140625 0.006975446875000001 0.4189453140625 0.0104631671875 0.35546875 0.074986046875 0.3222656265625 0.137765065625 0.2734375 0.3051757796875 0.2441406265625 0.577218190625 0.25 0.706263953125 0.2958984359375 0.854492190625 0.3818359359375 0.95912388125 0.4863281265625 0.9887695296875 0.5712890640625 0.9730747734375 0.6416015640625 0.9172712031250001 0.6894531265625 0.8283342625 0.73828125 0.6399972078125


================================================
FILE: TumorDetection/train/labels/no_tumor_83_jpg.rf.bb18a612f96e88b50dde87c657ec7e3c.txt
================================================
0 0.7954846390625 0.24609375 0.670352225 0.16015625 0.5769994703125 0.177734375 0.5333024359375 0.1484375 0.4776880296875 0.166015625 0.4141287078125 0.154296875 0.3624867578125 0.16015625 0.2413268015625 0.2109375 0.16287076250000002 0.2900390625 0.1390360171875 0.3408203125 0.135063559375 0.4052734375 0.144994703125 0.4619140625 0.1718087921875 0.490234375 0.26714777500000003 0.53125 0.366459215625 0.513671875 0.42902542343750005 0.4677734375 0.42902542343750005 0.4248046875 0.39029396250000004 0.435546875 0.3992319921875 0.4248046875 0.3803628171875 0.4140625 0.412142478125 0.40625 0.4161149359375 0.41796875 0.47172934375000003 0.369140625 0.5213850640625 0.375 0.5660752125 0.4228515625 0.5561440671875 0.4443359375 0.6375794484374999 0.5771484375 0.613744703125 0.5986328125 0.6157309328125 0.6357421875 0.6812764828125 0.6650390625 0.6683659953125 0.67578125 0.7070974578125 0.6767578125 0.766684321875 0.6220703125 0.846133475 0.5048828125 0.8441472453125 0.3466796875 0.7954846390625 0.24609375


================================================
FILE: TumorDetection/train/labels/no_tumor_867_jpg.rf.3dbcb4d0ee6596fd6d8ab68c5415a5ac.txt
================================================
0 0.7428850437500001 0.7822265625 0.76171875 0.7275390625 0.7993861609375 0.6591796875 0.8286830359374999 0.5478515625 0.8328683031249999 0.4248046875 0.8056640640625 0.3056640625 0.784737721875 0.2763671875 0.7742745531249999 0.2451171875 0.7596261156249999 0.2314453125 0.7491629468750001 0.1982421875 0.7062639500000001 0.1582031265625 0.6330217640625 0.10351562656249999 0.5890764500000001 0.0859375 0.430036271875 0.08203125 0.325404575 0.1171875 0.27622767812500004 0.1572265625 0.2260044640625 0.2255859375 0.2071707578125 0.2822265625 0.19461495625 0.2958984375 0.16950334843749998 0.3564453125 0.1611328140625 0.4267578125 0.163225446875 0.5478515625 0.1715959828125 0.5966796875 0.21554129375 0.7177734375 0.21763392812500001 0.7412109375 0.27622767812500004 0.8388671875 0.3274972109375 0.8847656265625 0.36516462031249997 0.9042968734375 0.44049944218750003 0.9277343734375 0.498046875 0.9306640625 0.5681501109375 0.9238281265625 0.6434849328125 0.8925781265625 0.6863839281249999 0.8564453125 0.7428850437500001 0.7822265625


================================================
FILE: TumorDetection/train/labels/no_tumor_871_jpg.rf.87ed23d46cc3aac75b4492934b5fcc70.txt
================================================
0 0.9152404171875 0.4619140625 0.87352005625 0.3251953125 0.8031169484375 0.1962890625 0.756181540625 0.1533203125 0.6062489953125 0.08203125156249999 0.40286223125 0.072265625 0.2698785828125 0.1289062515625 0.17991905625 0.2041015640625 0.09908585625 0.3720703125 0.078225675 0.5478515640625 0.135591175 0.7236328109375 0.21381685 0.8466796875 0.2881312421875 0.8984374984375 0.37157196406250004 0.9296874984375 0.5527947828124999 0.9404296875 0.68186715 0.912109375 0.7822567625 0.8427734359375 0.9022028046875 0.6162109375 0.9152404171875 0.4619140625


================================================
FILE: TumorDetection/train/labels/no_tumor_877_jpg.rf.6851911587ed126a7e017d207218587b.txt
================================================
0 0.8613281234375 0.6055041 0.83203125 0.3818403578125 0.7421875 0.172281534375 0.7001953109375 0.12291430781249998 0.6201171890625 0.0765695703125 0.5087890609375 0.06447963749999999 0.3720703109375 0.10074943437499999 0.27734375 0.18840144375 0.2011718765625 0.46042491406249997 0.1953125 0.5793092484375 0.2207031234375 0.7667031953125 0.2910156234375 0.8976774609374999 0.3896484390625 0.9671945703124999 0.5234375 0.98835195 0.6630859390625 0.9530896468750001 0.7578125 0.8795425609375 0.84765625 0.7364783671875 0.8613281234375 0.6055041


================================================
FILE: TumorDetection/train/labels/no_tumor_880_jpg.rf.c42192f2b2f546c0b1da245173dd798a.txt
================================================
0 0.914761325 0.3388671859375 0.8547430453124999 0.1767578140625 0.8154207296875 0.12011718593750001 0.7522980593749999 0.07226562343750001 0.6115655453125 0.023437498437500003 0.41288435312500005 0.023437498437500003 0.259734265625 0.06640625156249999 0.14280210625 0.1787109390625 0.0786446375 0.3486328140625 0.080714234375 0.5498046890625 0.1448717015625 0.7275390609375 0.263873459375 0.8496093765625 0.375631628125 0.8945312515625 0.525677321875 0.8994140609375 0.6881405875 0.8515625015625 0.8216295140625001 0.7236328140625 0.898204559375 0.5263671859375 0.914761325 0.3388671859375


================================================
FILE: TumorDetection/train/labels/no_tumor_881_jpg.rf.e4296a6bf08b8dae6ca3436cdbff68c4.txt
================================================
0 0.9926636375 0.2919921875 0.9463230749999999 0.1669921875 0.82071575625 0.041015625 0.662182253125 0 0.36706603437499996 0.0019531234375000002 0.2207274171875 0.041015625 0.0878031734375 0.1357421875 0.0317067015625 0.3369140640625 0.048779542187500004 0.5712890640625 0.12682680625 0.7568359359375 0.2207274171875 0.853515625 0.3816999 0.921875 0.5658426609375 0.9345703109375 0.752424403125 0.8828125015625 0.8951045578124999 0.7802734375 0.9829077265624999 0.5791015625 0.9926636375 0.2919921875


================================================
FILE: TumorDetection/train/labels/no_tumor_904_jpg.rf.73f0ee7d9426dddcb49489d5e4676928.txt
================================================
0 0.7109375 0.7412109359375 0.7734375 0.6240234359375 0.7832031265625 0.4775390640625 0.7402343734375 0.3232421859375 0.6572265640625 0.2324218734375 0.5361328140625 0.1777343734375 0.3896484359375 0.1933593734375 0.2822265640625 0.2714843734375 0.2402343734375 0.3505859359375 0.2109375 0.4951171859375 0.203125 0.6005859359375 0.2265625 0.6787109359375 0.3076171859375 0.7988281265625 0.3642578140625 0.8496093734375 0.4140625 0.8623046859375 0.5068359359375 0.8359375 0.5576171859375 0.8574218734375 0.6083984359375 0.84765625 0.7109375 0.7412109359375


================================================
FILE: TumorDetection/train/labels/no_tumor_914_jpg.rf.6714544aee2bde5213fd2c366dff62aa.txt
================================================
0 0.9874218734375001 0.3837890609375 0.9152343734375 0.2275390609375 0.7695703140625 0.07421874687499999 0.6123046859375 0.0136718734375 0.411210940625 0.017578126562499997 0.2823046859375 0.0605468734375 0.18562499999999998 0.12402343906249999 0.0721875 0.2763671875 0.010312499999999999 0.4326171875 0.012890626562499998 0.7646484390625 0.108281253125 0.8798828125 0.1894921859375 0.9355468734375 0.3570703140625 0.9902343734375 0.592968746875 0.9951171875 0.7850390593750001 0.9394531265625 0.9461718734375 0.8212890609375 0.9796875 0.7001953125 0.9874218734375001 0.3837890609375
0 0.9874218734375001 0.3837890609375 0.9152343734375 0.2275390609375 0.7695703140625 0.07421874687499999 0.6123046859375 0.0136718734375 0.411210940625 0.017578126562499997 0.2823046859375 0.0605468734375 0.18562499999999998 0.12402343906249999 0.0721875 0.2763671875 0.010312499999999999 0.4326171875 0.012890626562499998 0.7646484390625 0.108281253125 0.8798828125 0.1894921859375 0.9355468734375 0.3570703140625 0.9902343734375 0.592968746875 0.9951171875 0.7850390593750001 0.9394531265625 0.9461718734375 0.8212890609375 0.9796875 0.7001953125 0.9874218734375001 0.3837890609375


================================================
FILE: TumorDetection/train/labels/no_tumor_915_jpg.rf.9b3b8ca77f41c0d77cd67d05d4b39bf1.txt
================================================
0 0.5732421875 0.8984375 0.6083984375 0.8828125 0.6943359375 0.8203125 0.7421875 0.7490234375 0.7734375 0.6708984375 0.791015625 0.5966796875 0.791015625 0.5498046875 0.783203125 0.5361328125 0.7890625 0.4990234375 0.787109375 0.4560546875 0.7460937484375 0.2900390625 0.7226562515625 0.2353515625 0.6914062515625 0.1845703125 0.6591796875 0.150390625 0.5595703125 0.115234375 0.5126953125 0.1328125 0.5 0.1513671875 0.4990234375 0.173828125 0.4921875 0.1474609375 0.4794921875 0.1328125 0.4677734375 0.125 0.4345703125 0.125 0.4052734375 0.134765625 0.3203125 0.1806640625 0.2773437484375 0.2353515625 0.267578125 0.2646484375 0.25 0.2841796875 0.2421875 0.3974609375 0.232421875 0.4423828125 0.2148437484375 0.4755859375 0.208984375 0.5009765625 0.2109375 0.5322265625 0.216796875 0.5400390625 0.208984375 0.6044921875 0.212890625 0.6494140625 0.2265625 0.6728515625 0.234375 0.7099609375 0.2890625 0.8037109375 0.3310546875 0.845703125 0.3583984375 0.857421875 0.3779296875 0.8828125 0.4521484375 0.904296875 0.4765625 0.9033203125 0.4921875 0.8896484375 0.5048828125 0.8554687484375 0.5361328125 0.8984375 0.5732421875 0.8984375


================================================
FILE: TumorDetection/train/labels/no_tumor_920_jpg.rf.2b10abebde1cf6bca35139c4fe34478c.txt
================================================
0 0.8427734359375 0.857421875 0.896484375 0.7958984359375 0.9414062484375 0.7216796875 0.9726562484375 0.6435546875 0.982421875 0.5654296875 0.974609375 0.3876953125 0.9414062484375 0.2939453125 0.896484375 0.2216796875 0.8330078125 0.1523437515625 0.7216796875 0.07421875156249999 0.6318359359375 0.035156248437499996 0.5615234359375 0.0234375 0.4462890640625 0.0234375 0.3603515640625 0.0390625 0.2607421875 0.08203124843750001 0.2255859359375 0.111328125 0.1982421875 0.12109375156249999 0.087890625 0.2431640640625 0.042968751562500004 0.3330078125 0.015625 0.4501953125 0.013671875 0.5556640640625 0.041015625 0.6748046875 0.076171875 0.7470703125 0.1484375 0.8466796875 0.1806640640625 0.875 0.1982421875 0.8789062484375 0.2470703125 0.927734375 0.3095703125 0.9570312484375 0.4130859359375 0.9882812484375 0.4882812484375 0.9892578125 0.5576171875 0.9882812484375 0.6669921875 0.966796875 0.7568359359375 0.9257812484375 0.8427734359375 0.857421875


================================================
FILE: TumorDetection/train/labels/no_tumor_932_jpg.rf.a6dc07dc604c03e84f7224782a6cdce9.txt
================================================
0 0.9629304843750001 0.3720703109375 0.909149790625 0.1982421875 0.7798200359374999 0.08789062656249999 0.618477959375 0.0273437515625 0.380306321875 0.019531248437500003 0.23689114218750001 0.07031249843750001 0.094756459375 0.1826171875 0.046097734375 0.2880859375 0.030731826562499998 0.4912109375 0.0768295609375 0.6845703109375 0.17670798906250001 0.8642578125 0.3239646421875 0.9589843734375 0.41872110156250003 0.9804687515625 0.5480508625 0.9814453109375 0.7183563875 0.9414062484375 0.8502471328125001 0.8408203109375 0.947564571875 0.6474609375 0.9629304843750001 0.3720703109375


================================================
FILE: TumorDetection/train/labels/no_tumor_934_jpg.rf.d1cd0942b91c29942b34b7693c96adb8.txt
================================================
0 0.81640625 0.7255859359375 0.84765625 0.6240234359375 0.8457031265625 0.5107421859375 0.7578125 0.3505859359375 0.7441406265625 0.2353515640625 0.6943359359375 0.19140625 0.6025390640625 0.1542968734375 0.5576171859375 0.1542968734375 0.5341796859375 0.171875 0.5048828140625 0.1542968734375 0.4541015640625 0.15625 0.3564453140625 0.2011718734375 0.3125 0.2568359359375 0.3027343734375 0.3486328140625 0.2167968734375 0.5283203140625 0.2207031265625 0.6005859359375 0.265625 0.6572265640625 0.2597656265625 0.7099609359375 0.2988281265625 0.7783203140625 0.3681640640625 0.8359375 0.4667968734375 0.8603515640625 0.5302734359375 0.8359375 0.6318359359375 0.8574218734375 0.7392578140625 0.8046875 0.81640625 0.7255859359375


================================================
FILE: TumorDetection/train/labels/no_tumor_960_jpg.rf.f968706e3f72df5eb006f51fcdec01b0.txt
================================================
0 0.9521484359375 0.6630859375 0.96435546875 0.4775390625 0.9350585953125 0.3408203125 0.8642578140625 0.1982421890625 0.8032226546875 0.1220703125 0.7287597640625 0.078125 0.621337890625 0.042968748437499996 0.465087890625 0.033203123437499996 0.3234863265625 0.0605468765625 0.2453613265625 0.10156250156249999 0.1904296859375 0.1513671890625 0.11230468593750001 0.2880859375 0.0610351546875 0.4326171890625 0.0610351546875 0.6396484359375 0.12939453125 0.7978515640625 0.191650390625 0.8632812515625 0.3845214859375 0.9550781234375 0.546875 0.9677734359375 0.6506347640625 0.9550781234375 0.7531738265625 0.9179687484375 0.8642578140625 0.8388671890625 0.9521484359375 0.6630859375


================================================
FILE: TumorDetection/train/labels/no_tumor_973_jpg.rf.090d284c0592a1cfe32f4348c26d126c.txt
================================================
0 0.7900355875 0.8349609375 0.8229537359375 0.7841796875 0.8599866546875001 0.6728515625 0.8538145015625 0.5244140625 0.8373554265625 0.4150390625 0.8064946624999999 0.3291015625 0.8023798937500001 0.2998046875 0.750945284375 0.2138671875 0.70876890625 0.16796875 0.66762121875 0.14453125 0.6223587640625 0.1328125 0.5956127671875 0.11328125 0.5503503109375 0.123046875 0.52154693125 0.15234375 0.5164034703125 0.1318359375 0.49891570312500005 0.119140625 0.45776801562499997 0.11328125 0.35695618281249997 0.15625 0.303464190625 0.1953125 0.2859764234375 0.2138671875 0.2715747328125 0.2548828125 0.24688612031249998 0.2841796875 0.24071396875 0.3505859375 0.23248443124999998 0.3759765625 0.21602535625 0.3896484375 0.205738434375 0.4345703125 0.205738434375 0.4736328125 0.19545151249999998 0.5205078125 0.197508896875 0.6474609375 0.213967971875 0.6708984375 0.209853203125 0.7021484375 0.23248443124999998 0.7548828125 0.2859764234375 0.8369140625 0.3404971078125 0.880859375 0.4598254 0.923828125 0.515374778125 0.919921875 0.5400633890625001 0.89453125 0.5729815390625 0.92578125 0.6275022234375001 0.9306640625 0.653219528125 0.916015625 0.6964246000000001 0.908203125 0.7643182828125 0.861328125 0.7900355875 0.8349609375


================================================
FILE: TumorDetection/train/labels/no_tumor_97_jpg.rf.932de25b4dfeb8a7b4bab146c57b1a3b.txt
================================================
0 0.8320312515625 0.7333984375 0.8398437484375 0.6708984375 0.8525390625 0.6601562515625 0.861328125 0.6669921890625 0.8515625 0.6728515625 0.8525390625 0.7109375 0.8740234375 0.703125 0.8789062515625 0.7119140625 0.880859375 0.6982421890625 0.8710937484375 0.6962890625 0.8867187484375 0.6748046890625 0.837890625 0.6455078109375 0.833984375 0.5986328109375 0.8476562515625 0.5732421890625 0.8476562515625 0.5400390625 0.833984375 0.4150390625 0.740234375 0.2119140625 0.6826171890625 0.166015625 0.5771484375 0.1328125 0.544921875 0.0322265625 0.5205078109375 0.013671875 0.4873046890625 0.011718748437499999 0.4570312515625 0.034179689062500004 0.423828125 0.1318359375 0.3017578109375 0.189453125 0.265625 0.2197265625 0.228515625 0.2783203109375 0.201171875 0.3447265625 0.171875 0.5869140625 0.197265625 0.6123046890625 0.1875 0.6630859375 0.1992187484375 0.6728515625 0.216796875 0.7724609375 0.2539062515625 0.8486328109375 0.3134765625 0.900390625 0.4033203109375 0.951171875 0.4580078109375 0.9648437484375 0.541015625 0.9658203109375 0.6240234375 0.951171875 0.7021484375 0.9023437484375 0.810546875 0.7880859375 0.8320312515625 0.7333984375


================================================
FILE: TumorDetection/train/labels/no_tumor_982_jpg.rf.63ad40d046a68124bca367b4e8d111d3.txt
================================================
0 0.8378906265625 0.6337890625 0.8398437515625 0.4951171859375 0.7851562484375 0.2744140625 0.7109375015625 0.1552734359375 0.5849609375 0.07421875156249999 0.4384765640625 0.06835937343750001 0.3076171859375 0.1445312484375 0.2304687515625 0.2451171859375 0.1679687515625 0.3974609375 0.1601562484375 0.5849609375 0.201171875 0.7783203109375 0.2958984359375 0.8808593734375 0.3720703109375 0.9277343734375 0.546875 0.9443359375 0.6181640625 0.935546875 0.7060546890625 0.8945312484375 0.8027343734375 0.7939453109375 0.8378906265625 0.6337890625


================================================
FILE: TumorDetection/train/labels/no_tumor_992_jpg.rf.9c52cd6b6948f0c7d4731a65ec2350b5.txt
================================================
0 0.8613281234375 0.621624009375 0.83203125 0.3818403578125 0.75390625 0.1904164296875 0.7197265609375 0.14104920781250002 0.6435546890625 0.08865950156249999 0.4951171890625 0.066494625 0.3720703109375 0.10074943437499999 0.27734375 0.18840144375 0.2011718765625 0.4624399046875 0.1953125 0.5793092484375 0.2207031234375 0.7626732171875 0.2871093765625 0.8916324937500001 0.3876953109375 0.965179584375 0.5253906234375 0.98835195 0.6611328109375 0.9530896468750001 0.7636718765625 0.873497596875 0.85546875 0.72035845625 0.8613281234375 0.621624009375


================================================
FILE: TumorDetection/train/labels/no_tumor_9_jpg.rf.f12a0c01706e830ae6441bb9eab92796.txt
================================================
0 0.5785544515625001 0.875 0.6591620390624999 0.853515625 0.72416815625 0.8242187515625 0.8216773343750001 0.7568359359375 0.8346785578125001 0.7158203125 0.8762824734375 0.6513671875 0.8866834515625 0.5751953125 0.8710819843750001 0.5341796875 0.8788827171875001 0.4443359359375 0.8450795359375001 0.3212890640625 0.824277578125 0.2978515640625 0.8216773343750001 0.2431640640625 0.7618717046875 0.1806640640625 0.6929652203125001 0.1328125 0.6539615484375 0.1171875 0.50834784375 0.109375 0.47454466249999994 0.125 0.46154343906250006 0.111328125 0.435540990625 0.109375 0.3315312015625 0.123046875 0.30032826562500003 0.142578125 0.263924840625 0.146484375 0.21842055781249997 0.1689453125 0.17681664218749998 0.2216796875 0.15341443906250002 0.2744140640625 0.158614928125 0.3212890640625 0.14561370468749998 0.3427734359375 0.156014684375 0.3583984359375 0.122211503125 0.4052734359375 0.1066100328125 0.4736328125 0.11701101249999998 0.6298828125 0.14561370468749998 0.7060546875 0.1794168859375 0.7373046875 0.1950183546875 0.7744140640625 0.27432581875 0.830078125 0.32893095781250004 0.8476562484375 0.35493340624999997 0.8671875 0.42904037968750003 0.8779296875 0.4537427046875 0.875 0.48494564218749997 0.849609375 0.5161485781249999 0.875 0.5785544515625001 0.875


================================================
FILE: TumorDetection/train/labels/pituitary_1010_jpg.rf.4b32ede821cc21169ee7ee87c7475f8b.txt
================================================
3 0.5166015625 0.400390625 0.466796875 0.4169921875 0.46875 0.4462890625 0.4794921875 0.458984375 0.49609375 0.4619140625 0.5224609375 0.455078125 0.53125 0.4423828125 0.529296875 0.4130859375 0.5166015625 0.400390625


================================================
FILE: TumorDetection/train/labels/pituitary_1024_jpg.rf.3109ab187ccbc49368a10fad74be2637.txt
================================================
3 0.5556640625 0.341796875 0.5361328125 0.349609375 0.50390625 0.3896484375 0.5244140625 0.4296875 0.5703125 0.4423828125 0.59375 0.4208984375 0.59375 0.3583984375 0.5830078125 0.341796875 0.5556640625 0.341796875


================================================
FILE: TumorDetection/train/labels/pituitary_1027_jpg.rf.61b9436cbddca07208253f7fb77dea3c.txt
================================================
3 0.46484375 0.3564453125 0.453125 0.3994140625 0.4931640625 0.423828125 0.50390625 0.4228515625 0.537109375 0.3740234375 0.5185546875 0.34765625 0.4912109375 0.34375 0.46484375 0.3564453125


================================================
FILE: TumorDetection/train/labels/pituitary_1030_jpg.rf.f6c51f77adf6afb8cc8fd9e6ec491e09.txt
================================================
3 0.5419921875 0.443359375 0.4951171875 0.421875 0.45703125 0.4521484375 0.451171875 0.4951171875 0.458984375 0.4951171875 0.4931640625 0.48046875 0.5556640625 0.486328125 0.560546875 0.4736328125 0.5419921875 0.443359375


================================================
FILE: TumorDetection/train/labels/pituitary_1033_jpg.rf.f3674beb04a997d22e70560719dad517.txt
================================================
3 0.4736328125 0.427734375 0.462890625 0.4462890625 0.466796875 0.4599609375 0.4775390625 0.470703125 0.501953125 0.4697265625 0.515625 0.4599609375 0.517578125 0.4443359375 0.4833984375 0.42578125 0.4736328125 0.427734375


================================================
FILE: TumorDetection/train/labels/pituitary_1038_jpg.rf.d8fa0cf57747c0fa5270fce9ff9003cd.txt
================================================
3 0.5048828125 0.322265625 0.4384765625 0.32421875 0.400390625 0.3720703125 0.39453125 0.4462890625 0.4423828125 0.484375 0.50390625 0.4931640625 0.5302734375 0.486328125 0.5546875 0.4580078125 0.552734375 0.3662109375 0.5048828125 0.322265625


================================================
FILE: TumorDetection/train/labels/pituitary_1039_jpg.rf.da3dee214dabb24e4dd95db2b53ec015.txt
================================================
3 0.4912109375 0.400390625 0.47265625 0.4150390625 0.46484375 0.4287109375 0.4638671875 0.470703125 0.5390625 0.4755859375 0.556640625 0.4619140625 0.560546875 0.4443359375 0.5458984375 0.40625 0.4912109375 0.400390625


================================================
FILE: TumorDetection/train/labels/pituitary_103_jpg.rf.bff7d4ea3ad4757eb75a3204a97b2285.txt
================================================
3 0.4931640625 0.369140625 0.4658203125 0.37109375 0.44921875 0.3857421875 0.44921875 0.4130859375 0.435546875 0.4404296875 0.435546875 0.4560546875 0.4462890625 0.46875 0.482421875 0.4814453125 0.5126953125 0.47265625 0.533203125 0.4521484375 0.52734375 0.3974609375 0.4931640625 0.369140625


================================================
FILE: TumorDetection/train/labels/pituitary_1042_jpg.rf.1c6a9f22d4656dc9f913e1321233be2b.txt
================================================
3 0.576171875 0.4189453125 0.5703125 0.4189453125 0.5693359375 0.404296875 0.568359375 0.4228515625 0.5615234375 0.42578125 0.560546875 0.3876953125 0.572265625 0.3681640625 0.5341796875 0.3359375 0.5146484375 0.326171875 0.4716796875 0.326171875 0.404296875 0.3583984375 0.3984375 0.4130859375 0.4208984375 0.458984375 0.5380859375 0.447265625 0.57421875 0.4599609375 0.5859375 0.3759765625 0.5830078125 0.369140625 0.572265625 0.4091796875 0.576171875 0.4189453125


================================================
FILE: TumorDetection/train/labels/pituitary_1061_jpg.rf.73c2839491aecd2d6740e17c1e01a08a.txt
================================================
3 0.5322265625 0.38671875 0.515625 0.4072265625 0.51953125 0.4326171875 0.5341796875 0.44921875 0.5625 0.4501953125 0.578125 0.4404296875 0.58203125 0.3974609375 0.5556640625 0.384765625 0.5322265625 0.38671875


================================================
FILE: TumorDetection/train/labels/pituitary_1064_jpg.rf.c806771afa32d21b6cc68e8707be8385.txt
================================================
3 0.5419921875 0.45703125 0.4677734375 0.458984375 0.4580078125 0.44921875 0.4521484375 0.453125 0.4501953125 0.447265625 0.46875 0.4443359375 0.4462890625 0.42578125 0.42578125 0.4794921875 0.43359375 0.5341796875 0.4677734375 0.564453125 0.4814453125 0.56640625 0.4912109375 0.55859375 0.521484375 0.5654296875 0.56640625 0.5185546875 0.5703125 0.4931640625 0.564453125 0.4736328125 0.5419921875 0.45703125


================================================
FILE: TumorDetection/train/labels/pituitary_1066_jpg.rf.40ddf908fa6d7c77f89bb79611c83c1c.txt
================================================
3 0.560546875 0.3134765625 0.5419921875 0.294921875 0.5185546875 0.294921875 0.5087890625 0.302734375 0.4814453125 0.294921875 0.470703125 0.3056640625 0.474609375 0.3173828125 0.466796875 0.3447265625 0.451171875 0.3623046875 0.451171875 0.3779296875 0.4775390625 0.400390625 0.4951171875 0.392578125 0.501953125 0.4111328125 0.513671875 0.4169921875 0.5341796875 0.4140625 0.546875 0.3818359375 0.56640625 0.3642578125 0.560546875 0.3134765625


================================================
FILE: TumorDetection/train/labels/pituitary_1068_jpg.rf.edaf0b6b7d3b2985eda979ed6e841982.txt
================================================
3 0.5517578125 0.29296875 0.5244140625 0.287109375 0.5068359375 0.296875 0.498046875 0.3173828125 0.484375 0.3271484375 0.4765625 0.3662109375 0.4951171875 0.39453125 0.5146484375 0.40625 0.533203125 0.4052734375 0.5576171875 0.392578125 0.57421875 0.3720703125 0.572265625 0.3232421875 0.5517578125 0.29296875


================================================
FILE: TumorDetection/train/labels/pituitary_1078_jpg.rf.2cd1f412ca11bb69c6e684468ef5b211.txt
================================================
3 0.5751953125 0.54296875 0.6103515625 0.556640625 0.6240234375 0.5546875 0.6328125 0.5419921875 0.640625 0.4970703125 0.62109375 0.4794921875 0.61328125 0.4482421875 0.576171875 0.4287109375 0.580078125 0.4052734375 0.5576171875 0.390625 0.5341796875 0.390625 0.5009765625 0.40234375 0.4775390625 0.43359375 0.4326171875 0.43359375 0.435546875 0.4404296875 0.4150390625 0.451171875 0.384765625 0.4833984375 0.37109375 0.5087890625 0.375 0.5576171875 0.392578125 0.5615234375 0.4443359375 0.533203125 0.4755859375 0.484375 0.5166015625 0.48828125 0.5751953125 0.54296875


================================================
FILE: TumorDetection/train/labels/pituitary_1085_jpg.rf.ad625462177f73ddc314dd946a6292e2.txt
================================================
3 0.4833984375 0.416015625 0.462890625 0.4404296875 0.462890625 0.4580078125 0.4677734375 0.46484375 0.517578125 0.4697265625 0.537109375 0.4619140625 0.541015625 0.4287109375 0.5283203125 0.41796875 0.4833984375 0.416015625


================================================
FILE: TumorDetection/train/labels/pituitary_1087_jpg.rf.0b03feac0d2d2fc41b4f53e85d277900.txt
================================================
3 0.5341796875 0.380859375 0.4541015625 0.375 0.44921875 0.3896484375 0.435546875 0.3955078125 0.439453125 0.4033203125 0.431640625 0.4130859375 0.44140625 0.4189453125 0.439453125 0.4501953125 0.4482421875 0.462890625 0.5361328125 0.45703125 0.57421875 0.4677734375 0.57421875 0.4365234375 0.55859375 0.4248046875 0.556640625 0.3994140625 0.5341796875 0.380859375


================================================
FILE: TumorDetection/train/labels/pituitary_1089_jpg.rf.438fabf0bedd684773e3815fa8a7a8e3.txt
================================================
1 0.5185546875 0.373046875 0.4970703125 0.380859375 0.48046875 0.3974609375 0.474609375 0.4345703125 0.4990234375 0.45703125 0.53125 0.4619140625 0.546875 0.4501953125 0.5546875 0.4150390625 0.54296875 0.3837890625 0.5185546875 0.373046875


================================================
FILE: TumorDetection/train/labels/pituitary_1101_jpg.rf.81ca54811968652994121d9bcb03111a.txt
================================================
3 0.4541015625 0.37890625 0.4453125 0.3935546875 0.455078125 0.4365234375 0.4658203125 0.4453125 0.498046875 0.4443359375 0.521484375 0.4228515625 0.517578125 0.3935546875 0.4951171875 0.376953125 0.4541015625 0.37890625


================================================
FILE: TumorDetection/train/labels/pituitary_1111_jpg.rf.1e35a2f8c97d6330c6b2b4c6e7ca38d7.txt
================================================
3 0.5478515625 0.380859375 0.5259486609375 0.3655133921875 0.46735491093750003 0.3766741078125 0.484375 0.4658203125 0.4912109375 0.48046875 0.505859375 0.4814453125 0.5419921875 0.46484375 0.55859375 0.4462890625 0.5625 0.3994140625 0.5478515625 0.380859375
3 0.50390625 0.6728515625 0.513671875 0.6064453125 0.4951171875 0.599609375 0.48828125 0.6064453125 0.48828125 0.6650390625 0.48046875 0.6865234375 0.484375 0.7041015625 0.505859375 0.7119140625 0.5205078125 0.708984375 0.537109375 0.6845703125 0.5166015625 0.68359375 0.50390625 0.6728515625
3 0.6396484375 0.626953125 0.62890625 0.6376953125 0.634765625 0.6435546875 0.607421875 0.6689453125 0.6015625 0.6845703125 0.6220703125 0.7109375 0.642578125 0.7138671875 0.65625 0.6943359375 0.65625 0.6689453125 0.65234375 0.6357421875 0.6396484375 0.626953125
3 0.3681640625 0.634765625 0.35546875 0.6513671875 0.3515625 0.6962890625 0.359375 0.7236328125 0.37890625 0.7275390625 0.408203125 0.7001953125 0.40625 0.6630859375 0.3935546875 0.646484375 0.3681640625 0.634765625


================================================
FILE: TumorDetection/train/labels/pituitary_1124_jpg.rf.bd04ab69a51815d24906c9e24006d994.txt
================================================
3 0.564453125 0.5439453125 0.5546875 0.5107421875 0.5419921875 0.50390625 0.4697265625 0.51953125 0.4560546875 0.515625 0.443359375 0.5400390625 0.4833984375 0.583984375 0.521484375 0.5986328125 0.5478515625 0.59765625 0.5625 0.5830078125 0.55859375 0.5576171875 0.564453125 0.5439453125


================================================
FILE: TumorDetection/train/labels/pituitary_113_jpg.rf.9a22b4ebeb1132033df1f5184951f6eb.txt
================================================
3 0.4609375 0.4931640625 0.4404296875 0.46875 0.3994140625 0.466796875 0.37890625 0.4931640625 0.3828125 0.5615234375 0.4052734375 0.595703125 0.42578125 0.5986328125 0.45703125 0.5888671875 0.44921875 0.5341796875 0.458984375 0.5205078125 0.4609375 0.4931640625


================================================
FILE: TumorDetection/train/labels/pituitary_1148_jpg.rf.456fb9d350403fbe6095508093ddce49.txt
================================================
3 0.5576171875 0.515625 0.5439453125 0.52734375 0.5205078125 0.53125 0.4775390625 0.533203125 0.4638671875 0.52734375 0.427734375 0.5341796875 0.4326171875 0.5859375 0.4560546875 0.591796875 0.4638671875 0.58203125 0.4775390625 0.58203125 0.486328125 0.6044921875 0.4970703125 0.611328125 0.548828125 0.6240234375 0.5595703125 0.591796875 0.583984375 0.5908203125 0.5869140625 0.529296875 0.5576171875 0.515625


================================================
FILE: TumorDetection/train/labels/pituitary_114_jpg.rf.dab702f45daa231783befdf032cae55c.txt
================================================
3 0.5126953125 0.37109375 0.4775390625 0.361328125 0.443359375 0.3798828125 0.423828125 0.4384765625 0.431640625 0.4619140625 0.4501953125 0.484375 0.49609375 0.4912109375 0.54296875 0.4599609375 0.548828125 0.4267578125 0.54296875 0.4072265625 0.5126953125 0.37109375


================================================
FILE: TumorDetection/train/labels/pituitary_1177_jpg.rf.35da09d3f616eeda96cb4e7609a423db.txt
================================================
3 0.56640625 0.5771484375 0.5625 0.5478515625 0.529296875 0.5263671875 0.54296875 0.5068359375 0.544921875 0.4853515625 0.53125 0.4619140625 0.529296875 0.4169921875 0.5087890625 0.4140625 0.4765625 0.4365234375 0.4609375 0.4736328125 0.466796875 0.5166015625 0.478515625 0.5341796875 0.453125 0.5576171875 0.4453125 0.6298828125 0.4794921875 0.6640625 0.51953125 0.6630859375 0.5595703125 0.625 0.5791015625 0.6171875 0.58203125 0.6064453125 0.56640625 0.5771484375


================================================
FILE: TumorDetection/train/labels/pituitary_1181_jpg.rf.f46ac8ce835c7328a64bf0a3a7771e79.txt
================================================
3 0.537109375 0.4736328125 0.5048828125 0.455078125 0.4658203125 0.474609375 0.453125 0.4892578125 0.46875 0.5107421875 0.45703125 0.5322265625 0.4609375 0.5537109375 0.4716796875 0.560546875 0.5029296875 0.556640625 0.5361328125 0.583984375 0.5546875 0.5869140625 0.56640625 0.5810546875 0.576171875 0.5595703125 0.546875 0.5263671875 0.537109375 0.4736328125


================================================
FILE: TumorDetection/train/labels/pituitary_1188_jpg.rf.d22566a2b264e6995db3a783b303c8b9.txt
================================================
3 0.5947265625 0.580078125 0.61328125 0.5830078125 0.56640625 0.4794921875 0.5283203125 0.443359375 0.5068359375 0.435546875 0.4736328125 0.439453125 0.44140625 0.4716796875 0.443359375 0.5380859375 0.435546875 0.5537109375 0.435546875 0.5869140625 0.4541015625 0.611328125 0.482421875 0.6181640625 0.5146484375 0.60546875 0.5595703125 0.615234375 0.5771484375 0.60546875 0.5947265625 0.580078125


================================================
FILE: TumorDetection/train/labels/pituitary_11_jpg.rf.bde1d9fcca94f9c92d0378690a700e61.txt
================================================
3 0.54296875 0.4033203125 0.5673828125 0.41015625 0.5751953125 0.423828125 0.5966796875 0.4296875 0.6259765625 0.451171875 0.642578125 0.4501953125 0.6328125 0.4228515625 0.6025390625 0.400390625 0.5576171875 0.39453125 0.54296875 0.4033203125


================================================
FILE: TumorDetection/train/labels/pituitary_1206_jpg.rf.3d813be76058ce3c9d5727a3d008d368.txt
================================================
3 0.5126953125 0.5 0.4755859375 0.501953125 0.43359375 0.5302734375 0.439453125 0.5732421875 0.478515625 0.5830078125 0.5283203125 0.55859375 0.54296875 0.5380859375 0.53125 0.5322265625 0.5126953125 0.5


================================================
FILE: TumorDetection/train/labels/pituitary_1207_jpg.rf.339dcbf52551b285173535c6c18a040c.txt
================================================
3 0.4658203125 0.5 0.4384765625 0.5234375 0.41796875 0.5263671875 0.435546875 0.5498046875 0.4404296875 0.578125 0.490234375 0.5849609375 0.5224609375 0.568359375 0.541015625 0.5400390625 0.5126953125 0.4921875 0.4658203125 0.5


================================================
FILE: TumorDetection/train/labels/pituitary_1236_jpg.rf.d041da130c5eede6ef69dc356eb82620.txt
================================================
3 0.4951171875 0.486328125 0.46484375 0.4951171875 0.4638671875 0.52734375 0.4873046875 0.521484375 0.53515625 0.5361328125 0.529296875 0.5205078125 0.5390625 0.4990234375 0.5302734375 0.48828125 0.4951171875 0.486328125


================================================
FILE: TumorDetection/train/labels/pituitary_1238_jpg.rf.75ded478842a9b689f4c0ad285632be3.txt
================================================
3 0.4990234375 0.5703125 0.4638671875 0.578125 0.451171875 0.5908203125 0.447265625 0.6103515625 0.4677734375 0.634765625 0.49609375 0.6357421875 0.5517578125 0.626953125 0.556640625 0.5947265625 0.5322265625 0.578125 0.4990234375 0.5703125


================================================
FILE: TumorDetection/train/labels/pituitary_1242_jpg.rf.78cb91ce4d04f66626021099d15dc5bd.txt
================================================
3 0.53515625 0.4970703125 0.55078125 0.4482421875 0.5205078125 0.4296875 0.4990234375 0.4296875 0.47265625 0.4541015625 0.466796875 0.4951171875 0.443359375 0.5087890625 0.439453125 0.5322265625 0.4501953125 0.552734375 0.513671875 0.5576171875 0.5244140625 0.556640625 0.54296875 0.5380859375 0.546875 0.5146484375 0.53515625 0.4970703125


================================================
FILE: TumorDetection/train/labels/pituitary_124_jpg.rf.02166d4ed6c79fa7861917ce569a6ebd.txt
================================================
3 0.4189453125 0.568359375 0.3994140625 0.57421875 0.37890625 0.5966796875 0.373046875 0.6513671875 0.3935546875 0.6875 0.4189453125 0.701171875 0.435546875 0.7001953125 0.4609375 0.6728515625 0.4765625 0.6103515625 0.4638671875 0.576171875 0.4189453125 0.568359375


================================================
FILE: TumorDetection/train/labels/pituitary_1259_jpg.rf.e03d0d37b7d24d62f1085443a377b89d.txt
================================================
3 0.5546875 0.5458984375 0.55859375 0.5126953125 0.55078125 0.4931640625 0.5224609375 0.466796875 0.4892578125 0.458984375 0.439453125 0.4892578125 0.423828125 0.5419921875 0.458984375 0.5771484375 0.439453125 0.6240234375 0.47265625 0.6572265625 0.4775390625 0.673828125 0.5068359375 0.671875 0.509765625 0.6787109375 0.541015625 0.6494140625 0.548828125 0.6279296875 0.529296875 0.5712890625 0.5546875 0.5458984375


================================================
FILE: TumorDetection/train/labels/pituitary_1261_jpg.rf.492393a04066b17cdb6b60d080852e1d.txt
================================================
3 0.6123046875 0.658203125 0.626953125 0.6474609375 0.626953125 0.6298828125 0.60546875 0.5712890625 0.568359375 0.5498046875 0.5703125 0.5283203125 0.5546875 0.5009765625 0.5166015625 0.46484375 0.4912109375 0.4609375 0.4765625 0.4677734375 0.453125 0.5009765625 0.447265625 0.5517578125 0.4609375 0.5673828125 0.4248046875 0.580078125 0.404296875 0.6162109375 0.423828125 0.6357421875 0.427734375 0.6533203125 0.4521484375 0.671875 0.4814453125 0.685546875 0.5078125 0.6845703125 0.5517578125 0.671875 0.5732421875 0.673828125 0.5908203125 0.66015625 0.6123046875 0.658203125


================================================
FILE: TumorDetection/train/labels/pituitary_1279_jpg.rf.5515a4c82ff4beb3a384ae1ca177d8d1.txt
================================================
3 0.4599609375 0.474609375 0.4345703125 0.484375 0.423828125 0.5087890625 0.4453125 0.5205078125 0.455078125 0.5517578125 0.490234375 0.5576171875 0.5341796875 0.53515625 0.55078125 0.5009765625 0.4912109375 0.4765625 0.4599609375 0.474609375


================================================
FILE: TumorDetection/train/labels/pituitary_1282_jpg.rf.9ee76eab16909943dc2d105ea0669e10.txt
================================================
3 0.6025390625 0.533203125 0.5810546875 0.533203125 0.56640625 0.5517578125 0.568359375 0.5732421875 0.556640625 0.5810546875 0.564453125 0.6220703125 0.6171875 0.6298828125 0.6328125 0.6162109375 0.650390625 0.5810546875 0.650390625 0.5634765625 0.6025390625 0.533203125


================================================
FILE: TumorDetection/train/labels/pituitary_1299_jpg.rf.e0a6171844c4f1d06a5af3a6037b3ab6.txt
================================================
3 0.5576171875 0.466796875 0.5224609375 0.4765625 0.5 0.5107421875 0.5107421875 0.53515625 0.544921875 0.5400390625 0.5693359375 0.53125 0.59375 0.4970703125 0.5888671875 0.482421875 0.5576171875 0.466796875


================================================
FILE: TumorDetection/train/labels/pituitary_1309_jpg.rf.6bae60f8182afc580ca187c230fd12b7.txt
================================================
3 0.55859375 0.5478515625 0.55859375 0.5302734375 0.546875 0.5009765625 0.521484375 0.4794921875 0.5078125 0.4462890625 0.4892578125 0.44140625 0.4609375 0.4638671875 0.451171875 0.4912109375 0.435546875 0.5048828125 0.439453125 0.5263671875 0.427734375 0.5478515625 0.423828125 0.5830078125 0.4521484375 0.623046875 0.494140625 0.6318359375 0.5185546875 0.62890625 0.5390625 0.6083984375 0.544921875 0.5615234375 0.55859375 0.5478515625


================================================
FILE: TumorDetection/train/labels/pituitary_1317_jpg.rf.8c09c8a2c50efed5cd9390131e44288f.txt
================================================
3 0.541015625 0.5302734375 0.55078125 0.5224609375 0.55078125 0.5107421875 0.5439453125 0.5 0.498046875 0.4814453125 0.4892578125 0.4453125 0.474609375 0.4541015625 0.47265625 0.4873046875 0.41796875 0.5166015625 0.4140625 0.5478515625 0.44921875 0.6396484375 0.4697265625 0.650390625 0.5078125 0.6494140625 0.525390625 0.6357421875 0.541015625 0.6064453125 0.546875 0.5771484375 0.541015625 0.5302734375


================================================
FILE: TumorDetection/train/labels/pituitary_1342_jpg.rf.b792feb6aad2cba6c1326b4d6077fef9.txt
================================================
3 0.5576171875 0.548828125 0.556640625 0.5771484375 0.54296875 0.6005859375 0.55078125 0.6044921875 0.55078125 0.6337890625 0.576171875 0.6357421875 0.59765625 0.6220703125 0.60546875 0.5732421875 0.5849609375 0.55078125 0.5576171875 0.548828125


================================================
FILE: TumorDetection/train/labels/pituitary_1343_jpg.rf.abb4d43d5f4e56c6e6fafc98adcda5fa.txt
================================================
3 0.5751953125 0.541015625 0.5576171875 0.54296875 0.546875 0.5556640625 0.546875 0.5791015625 0.53125 0.6279296875 0.548828125 0.6357421875 0.5888671875 0.62890625 0.603515625 0.6005859375 0.59765625 0.5595703125 0.5751953125 0.541015625


================================================
FILE: TumorDetection/train/labels/pituitary_1346_jpg.rf.bd82a62829e4af57da80bc166d55a33d.txt
================================================
3 0.70703125 0.5732421875 0.669921875 0.5478515625 0.693359375 0.5380859375 0.6875 0.4931640625 0.666015625 0.4755859375 0.666015625 0.4580078125 0.650390625 0.4384765625 0.6201171875 0.423828125 0.5791015625 0.4296875 0.5380859375 0.453125 0.505859375 0.4775390625 0.490234375 0.5107421875 0.4921875 0.5498046875 0.5390625 0.5849609375 0.5390625 0.6240234375 0.5546875 0.6748046875 0.5712890625 0.69921875 0.59375 0.7041015625 0.6767578125 0.677734375 0.708984375 0.6416015625 0.71484375 0.6103515625 0.70703125 0.5732421875


================================================
FILE: TumorDetection/train/labels/pituitary_1353_jpg.rf.41935af18f13bf8b58005596af4179fc.txt
================================================
3 0.6533203125 0.423828125 0.6083984375 0.42578125 0.58203125 0.4521484375 0.5703125 0.4755859375 0.56640625 0.5087890625 0.572265625 0.5400390625 0.580078125 0.5634765625 0.5966796875 0.580078125 0.6357421875 0.595703125 0.658203125 0.5966796875 0.685546875 0.5654296875 0.6875 0.5009765625 0.67578125 0.4462890625 0.6533203125 0.423828125


================================================
FILE: TumorDetection/train/labels/pituitary_137_jpg.rf.9039b0515d9213947be053137f97cf98.txt
================================================
3 0.4619140625 0.578125 0.4267578125 0.576171875 0.375 0.6240234375 0.375 0.6630859375 0.4169921875 0.689453125 0.44140625 0.6884765625 0.4912109375 0.68359375 0.49609375 0.6552734375 0.48046875 0.5908203125 0.4619140625 0.578125


================================================
FILE: TumorDetection/train/labels/pituitary_1382_jpg.rf.2429a0c6bc4404c7c7647eb9144905db.txt
================================================
3 0.6435546875 0.439453125 0.5927734375 0.447265625 0.572265625 0.4638671875 0.552734375 0.4970703125 0.5712890625 0.52734375 0.623046875 0.5341796875 0.6669921875 0.52734375 0.681640625 0.4931640625 0.677734375 0.4599609375 0.6435546875 0.439453125


================================================
FILE: TumorDetection/train/labels/pituitary_1383_jpg.rf.c9d98fe5e7fea02b05ab0de87cc2f020.txt
================================================
3 0.6376953125 0.443359375 0.6064453125 0.4453125 0.5791015625 0.458984375 0.564453125 0.4814453125 0.56640625 0.4990234375 0.6240234375 0.53515625 0.638671875 0.5341796875 0.6416015625 0.5234375 0.673828125 0.5146484375 0.67578125 0.4599609375 0.6376953125 0.443359375


================================================
FILE: TumorDetection/train/labels/pituitary_1404_jpg.rf.dbe8debedafd2bfc0fe4803699dc829e.txt
================================================
3 0.626953125 0.5146484375 0.619140625 0.4814453125 0.6005859375 0.466796875 0.5712890625 0.466796875 0.529296875 0.4892578125 0.537109375 0.5185546875 0.552734375 0.5361328125 0.548828125 0.5888671875 0.580078125 0.5966796875 0.61328125 0.5673828125 0.6171875 0.5439453125 0.611328125 0.5361328125 0.626953125 0.5146484375


================================================
FILE: TumorDetection/train/labels/pituitary_1405_jpg.rf.f8c8c2496cdbaf8a83a97e5f4bcc8f1c.txt
================================================
3 0.615234375 0.5009765625 0.60546875 0.4794921875 0.5634765625 0.470703125 0.54296875 0.4951171875 0.548828125 0.5419921875 0.537109375 0.5654296875 0.5517578125 0.59375 0.583984375 0.5986328125 0.61328125 0.5791015625 0.623046875 0.5302734375 0.61328125 0.5263671875 0.615234375 0.5009765625


================================================
FILE: TumorDetection/train/labels/pituitary_1407_jpg.rf.09b2eaf7e619a9c235e3048c3bd3f0a6.txt
================================================
3 0.5380859375 0.490234375 0.4931640625 0.478515625 0.4697265625 0.484375 0.451171875 0.5068359375 0.44921875 0.5302734375 0.4609375 0.5595703125 0.43359375 0.6005859375 0.4775390625 0.6328125 0.51171875 0.6376953125 0.541015625 0.6044921875 0.53515625 0.5751953125 0.55859375 0.5244140625 0.5380859375 0.490234375


================================================
FILE: TumorDetection/train/labels/pituitary_1409_jpg.rf.db632f15e69b86603d45b0a408b638ae.txt
================================================
3 0.4443359375 0.365234375 0.466796875 0.3642578125 0.4208984375 0.341796875 0.3798828125 0.337890625 0.3173828125 0.36328125 0.3154296875 0.373046875 0.3076171875 0.36328125 0.2890625 0.3857421875 0.2939453125 0.3984375 0.306640625 0.3974609375 0.3583984375 0.375 0.3935546875 0.37109375 0.4248046875 0.380859375 0.4443359375 0.365234375


================================================
FILE: TumorDetection/train/labels/pituitary_1415_jpg.rf.3dcfacd6d61c498a37ae104a7b68484a.txt
================================================
3 0.5791015625 0.57421875 0.576171875 0.5966796875 0.5576171875 0.6015625 0.53515625 0.5791015625 0.533203125 0.5615234375 0.55859375 0.5361328125 0.5546875 0.5029296875 0.5146484375 0.4765625 0.4853515625 0.4765625 0.4638671875 0.484375 0.44921875 0.5009765625 0.443359375 0.5205078125 0.45703125 0.5556640625 0.42578125 0.5966796875 0.4365234375 0.6171875 0.4775390625 0.646484375 0.5078125 0.6455078125 0.5517578125 0.630859375 0.580078125 0.6005859375 0.58203125 0.5830078125 0.5791015625 0.57421875


================================================
FILE: TumorDetection/train/labels/pituitary_1421_jpg.rf.5b382942d4944afae4a203eb2ca6eeb8.txt
================================================
3 0.66796875 0.5654296875 0.6455078125 0.548828125 0.6064453125 0.556640625 0.58984375 0.5966796875 0.564453125 0.6220703125 0.5634765625 0.63671875 0.625 0.6669921875 0.646484375 0.6552734375 0.646484375 0.6181640625 0.671875 0.5888671875 0.66796875 0.5654296875


================================================
FILE: TumorDetection/train/labels/pituitary_1427_jpg.rf.5595b63eea9fb4a00d5581ae62ab47ec.txt
================================================
3 0.603515625 0.5966796875 0.63671875 0.5791015625 0.6484375 0.5517578125 0.646484375 0.5263671875 0.6181640625 0.505859375 0.5556640625 0.505859375 0.517578125 0.5439453125 0.51953125 0.5673828125 0.5390625 0.5869140625 0.5244140625 0.59375 0.513671875 0.6103515625 0.5078125 0.6591796875 0.5283203125 0.689453125 0.58984375 0.7080078125 0.615234375 0.6826171875 0.630859375 0.6416015625 0.626953125 0.6201171875 0.603515625 0.5966796875


================================================
FILE: TumorDetection/train/labels/pituitary_1436_jpg.rf.6fbbb593c8e651b7ba90ecc98d688f09.txt
================================================
3 0.64453125 0.5810546875 0.64453125 0.5595703125 0.6318359375 0.548828125 0.5947265625 0.541015625 0.580078125 0.5595703125 0.587890625 0.5791015625 0.5517578125 0.59375 0.54296875 0.6142578125 0.5791015625 0.673828125 0.619140625 0.6787109375 0.6298828125 0.67578125 0.642578125 0.6552734375 0.63671875 0.5927734375 0.64453125 0.5810546875


================================================
FILE: TumorDetection/train/labels/pituitary_1439_jpg.rf.52e40f4eac3531a59fe34d0037993d9e.txt
================================================
3 0.6328125 0.6357421875 0.64453125 0.5634765625 0.6357421875 0.548828125 0.6123046875 0.5390625 0.6005859375 0.5390625 0.58203125 0.5537109375 0.58203125 0.5673828125 0.55859375 0.5947265625 0.552734375 0.6240234375 0.560546875 0.6728515625 0.5810546875 0.6875 0.599609375 0.6884765625 0.6357421875 0.677734375 0.6484375 0.6650390625 0.646484375 0.6474609375 0.6328125 0.6357421875


================================================
FILE: TumorDetection/train/labels/pituitary_1451_jpg.rf.abba75cfa7ccfb3aeac67735105d4d6c.txt
================================================
3 0.53515625 0.5341796875 0.525390625 0.5166015625 0.5048828125 0.501953125 0.4794921875 0.5 0.4580078125 0.509765625 0.43359375 0.5458984375 0.435546875 0.6142578125 0.4482421875 0.626953125 0.47265625 0.6337890625 0.4931640625 0.625 0.5185546875 0.62890625 0.546875 0.6025390625 0.53515625 0.5341796875


================================================
FILE: TumorDetection/train/labels/pituitary_146_jpg.rf.fb709ef6cc597e964dddb0ca4824f553.txt
================================================
3 0.4580078125 0.482421875 0.4091796875 0.46875 0.3857421875 0.4765625 0.373046875 0.4931640625 0.3671875 0.5712890625 0.3828125 0.6083984375 0.4072265625 0.626953125 0.419921875 0.6259765625 0.4404296875 0.62109375 0.4609375 0.5986328125 0.478515625 0.5146484375 0.4580078125 0.482421875


================================================
FILE: TumorDetection/train/labels/pituitary_147_jpg.rf.d7addc2328755de0d735cd46731b550f.txt
================================================
3 0.5888671875 0.357421875 0.5693359375 0.3671875 0.5302734375 0.33984375 0.4990234375 0.34375 0.4609375 0.3740234375 0.44921875 0.4326171875 0.4375 0.4541015625 0.4501953125 0.470703125 0.521484375 0.4794921875 0.5810546875 0.466796875 0.609375 0.4365234375 0.59375 0.3623046875 0.5888671875 0.357421875


================================================
FILE: TumorDetection/train/labels/pituitary_14_jpg.rf.0ed43606cb64d0d8fb914beb9b3d71b1.txt
================================================
3 0.5498046875 0.359375 0.5263671875 0.349609375 0.4892578125 0.349609375 0.46484375 0.3662109375 0.4609375 0.3876953125 0.46875 0.4169921875 0.4794921875 0.427734375 0.494140625 0.4287109375 0.5361328125 0.42578125 0.55859375 0.4091796875 0.5625 0.3759765625 0.5498046875 0.359375


================================================
FILE: TumorDetection/train/labels/pituitary_152_jpg.rf.fee1aa85a22eb1514164cafefab9ef37.txt
================================================
3 0.548828125 0.6044921875 0.533203125 0.5810546875 0.533203125 0.5498046875 0.5107421875 0.53515625 0.4912109375 0.533203125 0.4658203125 0.546875 0.45703125 0.5615234375 0.4609375 0.5927734375 0.447265625 0.6123046875 0.453125 0.6630859375 0.4658203125 0.6796875 0.51953125 0.6884765625 0.537109375 0.6708984375 0.537109375 0.6552734375 0.51953125 0.6318359375 0.548828125 0.6044921875


================================================
FILE: TumorDetection/train/labels/pituitary_157_jpg.rf.fab3a62e15ce58bd6d8dcb1c462f48e7.txt
================================================
3 0.5625 0.5888671875 0.5224609375 0.568359375 0.5068359375 0.544921875 0.4892578125 0.54296875 0.474609375 0.5537109375 0.47265625 0.5732421875 0.44140625 0.5947265625 0.447265625 0.6025390625 0.447265625 0.6474609375 0.46875 0.6943359375 0.4814453125 0.70703125 0.5 0.7080078125 0.537109375 0.6826171875 0.544921875 0.6435546875 0.5625 0.6201171875 0.5625 0.5888671875


================================================
FILE: TumorDetection/train/labels/pituitary_158_jpg.rf.77c6fe774b3e1cddd6476bc3bedd447c.txt
================================================
3 0.57421875 0.4560546875 0.5810546875 0.46484375 0.580078125 0.4501953125 0.58984375 0.4345703125 0.5703125 0.4150390625 0.56640625 0.3759765625 0.5537109375 0.361328125 0.4970703125 0.35546875 0.455078125 0.3935546875 0.447265625 0.4482421875 0.484375 0.4755859375 0.4462890625 0.470703125 0.439453125 0.4794921875 0.447265625 0.5087890625 0.4375 0.5283203125 0.435546875 0.5595703125 0.4619140625 0.603515625 0.4873046875 0.615234375 0.515625 0.6162109375 0.5517578125 0.59375 0.583984375 0.5556640625 0.58984375 0.5244140625 0.57421875 0.4560546875


================================================
FILE: TumorDetection/train/labels/pituitary_16_jpg.rf.b84c67869927f85ff477ae7db6db9a5c.txt
================================================
3 0.443359375 0.5947265625 0.423828125 0.5498046875 0.4072265625 0.53515625 0.3818359375 0.533203125 0.3681640625 0.5390625 0.349609375 0.5654296875 0.359375 0.6220703125 0.349609375 0.6611328125 0.3701171875 0.6875 0.4140625 0.6962890625 0.4345703125 0.689453125 0.447265625 0.6748046875 0.451171875 0.6416015625 0.443359375 0.5947265625


================================================
FILE: TumorDetection/train/labels/pituitary_170_jpg.rf.1b460d77ac26e2c707ff8997576dd03b.txt
================================================
3 0.4306640625 0.541015625 0.3798828125 0.537109375 0.357421875 0.5556640625 0.349609375 0.5732421875 0.357421875 0.6123046875 0.3759765625 0.634765625 0.41796875 0.6357421875 0.447265625 0.6220703125 0.4453125 0.5751953125 0.435546875 0.5654296875 0.4306640625 0.541015625


================================================
FILE: TumorDetection/train/labels/pituitary_19_jpg.rf.13cadca08978ec32b69bed7772dd3ea1.txt
================================================
3 0.5693359359375 0.435546875 0.5126953125 0.392578125 0.4931640640625 0.3945312484375 0.4580078125 0.4179687515625 0.4208984359375 0.4140625 0.404296875 0.4208984359375 0.392578125 0.4580078125 0.4150390640625 0.5039062484375 0.4912109359375 0.525390625 0.5273437515625 0.5263671875 0.5693359359375 0.498046875 0.5859375 0.4755859359375 0.5859375 0.4580078125 0.5693359359375 0.435546875


================================================
FILE: TumorDetection/train/labels/pituitary_202_jpg.rf.b72bd3452c676440444e734d19732718.txt
================================================
3 0.5966796875 0.48828125 0.5576171875 0.46484375 0.53125 0.4619140625 0.5654296875 0.509765625 0.5888671875 0.5234375 0.607421875 0.5244140625 0.6396484375 0.517578125 0.658203125 0.5048828125 0.6357421875 0.48828125 0.5966796875 0.48828125
3 0.3642578125 0.4921875 0.34765625 0.5146484375 0.34765625 0.5263671875 0.3525390625 0.5390625 0.365234375 0.5400390625 0.4033203125 0.52734375 0.443359375 0.4794921875 0.3935546875 0.474609375 0.3642578125 0.4921875


================================================
FILE: TumorDetection/train/labels/pituitary_20_jpg.rf.5a0ebcd704a580a4392104b5d6d706f8.txt
================================================
3 0.603515625 0.6787109375 0.5966796875 0.625 0.5615234375 0.634765625 0.546875 0.6240234375 0.5380859375 0.59375 0.4970703125 0.58984375 0.4697265625 0.6015625 0.458984375 0.6220703125 0.44140625 0.6298828125 0.43359375 0.6474609375 0.4501953125 0.6875 0.5068359375 0.693359375 0.5576171875 0.685546875 0.603515625 0.7275390625 0.61328125 0.7158203125 0.603515625 0.6845703125 0.580078125 0.6943359375 0.603515625 0.6787109375


================================================
FILE: TumorDetection/train/labels/pituitary_21_jpg.rf.bf154eb052afca2a3503d6ac4e3809df.txt
================================================
3 0.591796875 0.48034399062500005 0.57421875 0.46069802031249996 0.578125 0.42533527187500003 0.5703125 0.39390171875 0.5361328125 0.35952126875 0.4990234375 0.34183989531250003 0.4560546875 0.34183989531250003 0.4345703125 0.35166288125 0.419921875 0.366397359375 0.40625 0.3978309140625 0.408203125 0.42729986875000003 0.43359375 0.484273184375 0.4296875 0.505883753125 0.4375 0.5176713359375 0.431640625 0.5216005296875 0.4384765625 0.5265120234375 0.4619140625 0.5265120234375 0.4912109375 0.5500871875 0.546875 0.549104890625 0.5732421875 0.54615799375 0.587890625 0.5196359328125 0.58203125 0.49606076874999994 0.591796875 0.48034399062500005


================================================
FILE: TumorDetection/train/labels/pituitary_243_jpg.rf.2207eb779fccb72755c361bc2ad5b425.txt
================================================
3 0.5048828125 0.5 0.4716796875 0.498046875 0.4541015625 0.5078125 0.43359375 0.5341796875 0.4296875 0.5810546875 0.4453125 0.5869140625 0.439453125 0.6181640625 0.453125 0.6220703125 0.5224609375 0.6171875 0.5390625 0.5986328125 0.54296875 0.5361328125 0.5048828125 0.5


================================================
FILE: TumorDetection/train/labels/pituitary_244_jpg.rf.58478c0560e03a47e4f680a098b10d0a.txt
================================================
3 0.529296875 0.5810546875 0.5087890625 0.552734375 0.462890625 0.5654296875 0.451171875 0.5888671875 0.44921875 0.6142578125 0.427734375 0.6474609375 0.4501953125 0.662109375 0.4697265625 0.6640625 0.4873046875 0.681640625 0.501953125 0.6806640625 0.54296875 0.6533203125 0.521484375 0.6318359375 0.529296875 0.6142578125 0.529296875 0.5810546875


================================================
FILE: TumorDetection/train/labels/pituitary_247_jpg.rf.3747d6cdc2b67c4ee7e8b5f82f23e8be.txt
================================================
3 0.564453125 0.4931640625 0.5625 0.4677734375 0.5546875 0.4521484375 0.5322265625 0.4375 0.5048828125 0.4375 0.4814453125 0.44921875 0.45703125 0.4931640625 0.462890625 0.5791015625 0.4794921875 0.59375 0.505859375 0.5966796875 0.5361328125 0.5859375 0.552734375 0.5693359375 0.548828125 0.5400390625 0.564453125 0.4931640625


================================================
FILE: TumorDetection/train/labels/pituitary_249_jpg.rf.3953374b4c05261b5e7f1ea7460238ff.txt
================================================
3 0.556640625 0.4794921875 0.53515625 0.4580078125 0.51953125 0.4267578125 0.4970703125 0.41796875 0.4765625 0.4365234375 0.462890625 0.4697265625 0.4453125 0.4814453125 0.462890625 0.5419921875 0.453125 0.5751953125 0.4736328125 0.60546875 0.5234375 0.6064453125 0.56640625 0.5966796875 0.56640625 0.5576171875 0.55078125 0.5244140625 0.556640625 0.4794921875


================================================
FILE: TumorDetection/train/labels/pituitary_254_jpg.rf.097b1debae13249fafc0b6f320034392.txt
================================================
3 0.53515625 0.4951171875 0.54296875 0.4814453125 0.5107421875 0.4453125 0.4873046875 0.439453125 0.4619140625 0.451171875 0.443359375 0.4716796875 0.4609375 0.5107421875 0.453125 0.5263671875 0.416015625 0.5556640625 0.416015625 0.5654296875 0.447265625 0.5849609375 0.4970703125 0.5546875 0.5244140625 0.556640625 0.53515625 0.5478515625 0.53515625 0.4951171875


================================================
FILE: TumorDetection/train/labels/pituitary_260_jpg.rf.6106032205be1c58d46f2b42244de930.txt
================================================
3 0.5888671875 0.6171875 0.59375 0.6357421875 0.599609375 0.6318359375 0.599609375 0.6005859375 0.5849609375 0.583984375 0.5693359375 0.583984375 0.564453125 0.5771484375 0.568359375 0.5576171875 0.595703125 0.5166015625 0.59375 0.4599609375 0.5673828125 0.423828125 0.5341796875 0.408203125 0.4833984375 0.41796875 0.44921875 0.4404296875 0.43359375 0.5029296875 0.4453125 0.5537109375 0.4384765625 0.56640625 0.412109375 0.5634765625 0.40234375 0.5751953125 0.396484375 0.6064453125 0.404296875 0.6435546875 0.41796875 0.6533203125 0.4384765625 0.689453125 0.501953125 0.7041015625 0.5693359375 0.69140625 0.58984375 0.6435546875 0.578125 0.6259765625 0.5888671875 0.6171875


================================================
FILE: TumorDetection/train/labels/pituitary_269_jpg.rf.9035e3b437756afc4755b01d129965b0.txt
================================================
3 0.576171875 0.4580078125 0.5556640625 0.435546875 0.5322265625 0.423828125 0.4912109375 0.42578125 0.4609375 0.4482421875 0.443359375 0.4853515625 0.451171875 0.5302734375 0.435546875 0.5439453125 0.419921875 0.5751953125 0.4208984375 0.583984375 0.439453125 0.5830078125 0.42578125 0.5869140625 0.4306640625 0.595703125 0.4521484375 0.59375 0.4677734375 0.60546875 0.48828125 0.6044921875 0.55859375 0.5791015625 0.560546875 0.5400390625 0.58203125 0.4912109375 0.576171875 0.4580078125


================================================
FILE: TumorDetection/train/labels/pituitary_277_jpg.rf.7575731b5aea84cd117d7395cddbe672.txt
================================================
3 0.4873046875 0.5 0.47265625 0.5263671875 0.45703125 0.5341796875 0.4609375 0.5478515625 0.5126953125 0.580078125 0.537109375 0.5830078125 0.5625 0.5693359375 0.56640625 0.5302734375 0.5205078125 0.5 0.4873046875 0.5


================================================
FILE: TumorDetection/train/labels/pituitary_280_jpg.rf.27a5e63dca0569f36558ddd3f4198e77.txt
================================================
3 0.423828125 0.6025390625 0.4453125 0.5927734375 0.4375 0.5888671875 0.4404296875 0.578125 0.4482421875 0.576171875 0.4541015625 0.591796875 0.4833984375 0.599609375 0.5556640625 0.572265625 0.568359375 0.5537109375 0.5625 0.5283203125 0.580078125 0.4951171875 0.580078125 0.4697265625 0.56640625 0.4443359375 0.5439453125 0.42578125 0.4931640625 0.423828125 0.453125 0.4560546875 0.443359375 0.4853515625 0.4453125 0.5263671875 0.41796875 0.5830078125 0.4140625 0.6337890625 0.43359375 0.6142578125 0.435546875 0.6064453125 0.4248046875 0.609375 0.423828125 0.6025390625


================================================
FILE: TumorDetection/train/labels/pituitary_295_jpg.rf.33f023252628dc40b0e4f044df06ba0b.txt
================================================
3 0.5419921875 0.58984375 0.5498046875 0.59765625 0.572265625 0.5888671875 0.5703125 0.5537109375 0.546875 0.5205078125 0.55859375 0.4892578125 0.556640625 0.4736328125 0.5400390625 0.458984375 0.5009765625 0.44921875 0.4580078125 0.44921875 0.43359375 0.4619140625 0.43359375 0.4912109375 0.44140625 0.4970703125 0.4326171875 0.5078125 0.4287109375 0.5 0.423828125 0.5029296875 0.4267578125 0.515625 0.44140625 0.5126953125 0.439453125 0.5400390625 0.455078125 0.5439453125 0.4375 0.5478515625 0.443359375 0.5595703125 0.4375 0.5751953125 0.453125 0.5791015625 0.451171875 0.6123046875 0.4423828125 0.611328125 0.44140625 0.6201171875 0.4638671875 0.63671875 0.5 0.6396484375 0.53125 0.6220703125 0.525390625 0.6162109375 0.5283203125 0.59375 0.5419921875 0.58984375


================================================
FILE: TumorDetection/train/labels/pituitary_2_jpg.rf.b59a4592aebe10effa30a561108513dc.txt
================================================
3 0.56640625 0.6533203125 0.5703125 0.6337890625 0.55078125 0.6083984375 0.54296875 0.5751953125 0.572265625 0.5517578125 0.576171875 0.5361328125 0.55859375 0.4853515625 0.5205078125 0.4609375 0.4970703125 0.45703125 0.455078125 0.4853515625 0.44140625 0.5087890625 0.439453125 0.5302734375 0.443359375 0.5458984375 0.470703125 0.5732421875 0.4453125 0.6220703125 0.453125 0.6611328125 0.4814453125 0.681640625 0.5078125 0.6884765625 0.5244140625 0.681640625 0.5458984375 0.65625 0.5615234375 0.662109375 0.56640625 0.6533203125


================================================
FILE: TumorDetection/train/labels/pituitary_314_jpg.rf.f9614bef9231de2a434382d7f7f5361b.txt
================================================
3 0.4892578125 0.623046875 0.453125 0.6416015625 0.439453125 0.6884765625 0.4619140625 0.708984375 0.49609375 0.7119140625 0.5205078125 0.705078125 0.537109375 0.6708984375 0.529296875 0.6435546875 0.4892578125 0.623046875


================================================
FILE: TumorDetection/train/labels/pituitary_339_jpg.rf.5bfb40b325964b3de4951e93b9097a44.txt
================================================
3 0.4619140625 0.49609375 0.4052734375 0.478515625 0.3759765625 0.490234375 0.34375 0.5244140625 0.3359375 0.5927734375 0.3662109375 0.65234375 0.3974609375 0.66796875 0.416015625 0.6669921875 0.4345703125 0.66015625 0.4609375 0.6337890625 0.474609375 0.5419921875 0.4619140625 0.49609375


================================================
FILE: TumorDetection/train/labels/pituitary_347_jpg.rf.5b5a6738b4564e26e2ac4a12a83fe71f.txt
================================================
3 0.5546875 0.4833984375 0.529296875 0.4638671875 0.52734375 0.4404296875 0.5126953125 0.41796875 0.4697265625 0.419921875 0.455078125 0.4384765625 0.4609375 0.4736328125 0.44140625 0.4970703125 0.4375 0.5166015625 0.4580078125 0.533203125 0.525390625 0.5380859375 0.5419921875 0.53515625 0.55078125 0.5166015625 0.56640625 0.5068359375 0.5546875 0.4833984375


================================================
FILE: TumorDetection/train/labels/pituitary_349_jpg.rf.f6d0644ef3211c291e4b7ccdb217c48b.txt
================================================
3 0.4404296875 0.458984375 0.4296875 0.4736328125 0.427734375 0.5244140625 0.4365234375 0.53515625 0.453125 0.5380859375 0.466796875 0.5244140625 0.46875 0.4853515625 0.4560546875 0.4609375 0.4404296875 0.458984375


================================================
FILE: TumorDetection/train/labels/pituitary_352_jpg.rf.3a2546e2cf83cbda2540be0c96ec5ab1.txt
================================================
3 0.53125 0.3701171875 0.4326171875 0.3671875 0.3828125 0.4072265625 0.3818359375 0.431640625 0.4033203125 0.419921875 0.4365234375 0.439453125 0.53125 0.3701171875


================================================
FILE: TumorDetection/train/labels/pituitary_353_jpg.rf.d4c4f71325bfc241e61864a4848251a5.txt
================================================
3 0.388671875 0.5712890625 0.38671875 0.6044921875 0.421875 0.6201171875 0.462890625 0.6025390625 0.4580078125 0.560546875 0.4091796875 0.556640625 0.388671875 0.5712890625


================================================
FILE: TumorDetection/train/labels/pituitary_362_jpg.rf.cbd51c6049ef39b3dafdcd32c7122718.txt
================================================
3 0.40625 0.5478515625 0.408203125 0.4951171875 0.3896484375 0.484375 0.3583984375 0.484375 0.3447265625 0.490234375 0.322265625 0.5185546875 0.322265625 0.5400390625 0.337890625 0.5634765625 0.337890625 0.6064453125 0.3583984375 0.63671875 0.3984375 0.6416015625 0.431640625 0.6240234375 0.439453125 0.5927734375 0.421875 0.5576171875 0.40625 0.5478515625


================================================
FILE: TumorDetection/train/labels/pituitary_366_jpg.rf.17aeda00d23ceea54c56aad7da13f09b.txt
================================================
3 0.4921875 0.5380859375 0.4765625 0.4853515625 0.4365234375 0.458984375 0.40234375 0.4716796875 0.3984375 0.5087890625 0.4453125 0.5537109375 0.4541015625 0.537109375 0.4775390625 0.5234375 0.4873046875 0.544921875 0.4921875 0.5380859375


================================================
FILE: TumorDetection/train/labels/pituitary_367_jpg.rf.6def2e69ca7ff611b14c49c62d099a33.txt
================================================
3 0.3759765625 0.51171875 0.376953125 0.5380859375 0.404296875 0.5986328125 0.4228515625 0.615234375 0.462890625 0.6142578125 0.470703125 0.5576171875 0.46484375 0.5361328125 0.4287109375 0.509765625 0.3759765625 0.51171875


================================================
FILE: TumorDetection/train/labels/pituitary_383_jpg.rf.8f63a70e496e5263199823752e7bc500.txt
================================================
3 0.4052734375 0.453125 0.380859375 0.4697265625 0.373046875 0.4892578125 0.375 0.5107421875 0.384765625 0.5380859375 0.3994140625 0.552734375 0.431640625 0.5595703125 0.4453125 0.5458984375 0.4453125 0.4755859375 0.4326171875 0.453125 0.4052734375 0.453125


================================================
FILE: TumorDetection/train/labels/pituitary_395_jpg.rf.6ef2fc99bbf408189d3afe1c223c6896.txt
================================================
3 0.4716796875 0.41796875 0.4404296875 0.404296875 0.4169921875 0.404296875 0.380859375 0.4326171875 0.375 0.4853515625 0.3935546875 0.5234375 0.4130859375 0.533203125 0.439453125 0.5322265625 0.4697265625 0.5234375 0.482421875 0.4990234375 0.478515625 0.4404296875 0.4716796875 0.41796875


================================================
FILE: TumorDetection/train/labels/pituitary_425_jpg.rf.657eea891f15b8c4358c3efe83a83aad.txt
================================================
3 0.5580357140625 0.56640625 0.5400390625 0.560546875 0.4990234375 0.5546875 0.4609375 0.5712890625 0.466796875 0.6005859375 0.46875 0.6026785703125 0.48130580468749995 0.5970982140625 0.48828125 0.6124441953125 0.5036272312500001 0.5998883937499999 0.5064174109375 0.6096540187499999 0.5203683046875 0.5970982140625 0.548828125 0.6064453125 0.57421875 0.6005859375 0.55766458125 0.5665038250000001 0.5580357140625 0.56640625


================================================
FILE: TumorDetection/train/labels/pituitary_429_jpg.rf.6820785128fc5fb59d697c85a276689e.txt
================================================
3 0.556640625 0.5244140625 0.544921875 0.4931640625 0.5146484375 0.4765625 0.4873046875 0.4765625 0.4521484375 0.49609375 0.44140625 0.5166015625 0.4453125 0.5400390625 0.46484375 0.5595703125 0.46484375 0.5791015625 0.4541015625 0.583984375 0.4423828125 0.56640625 0.4326171875 0.564453125 0.41796875 0.5810546875 0.41796875 0.5947265625 0.4541015625 0.634765625 0.517578125 0.6474609375 0.56640625 0.6142578125 0.5703125 0.5849609375 0.544921875 0.5595703125 0.556640625 0.5244140625


================================================
FILE: TumorDetection/train/labels/pituitary_434_jpg.rf.82eb679432f78789b1e9a4e1f979cb04.txt
================================================
3 0.54296875 0.5439453125 0.5634765625 0.537109375 0.578125 0.5205078125 0.5703125 0.5029296875 0.5478515625 0.48828125 0.4775390625 0.484375 0.4541015625 0.47265625 0.4306640625 0.482421875 0.41796875 0.5048828125 0.4248046875 0.51953125 0.4453125 0.5263671875 0.451171875 0.5517578125 0.443359375 0.5771484375 0.447265625 0.6025390625 0.4658203125 0.626953125 0.4921875 0.6396484375 0.53515625 0.6123046875 0.541015625 0.5908203125 0.533203125 0.5732421875 0.54296875 0.5439453125


================================================
FILE: TumorDetection/train/labels/pituitary_442_jpg.rf.08e48ca357ff70abbfa5bc1ce17f45a8.txt
================================================
3 0.5048828125 0.50390625 0.5107421875 0.51171875 0.525390625 0.4716796875 0.5126953125 0.4609375 0.4794921875 0.4609375 0.48046875 0.4462890625 0.4716796875 0.4375 0.4462890625 0.4375 0.453125 0.4287109375 0.4423828125 0.427734375 0.423828125 0.4404296875 0.435546875 0.4501953125 0.419921875 0.4716796875 0.427734375 0.5029296875 0.40625 0.5263671875 0.4609375 0.5673828125 0.4716796875 0.564453125 0.509765625 0.5224609375 0.4921875 0.5048828125 0.5048828125 0.50390625


================================================
FILE: TumorDetection/train/labels/pituitary_444_jpg.rf.616b3837eb98648e1c60e7920eae79bb.txt
================================================
3 0.5224609375 0.509765625 0.52734375 0.4892578125 0.4736328125 0.458984375 0.4404296875 0.45703125 0.3876953125 0.478515625 0.375 0.4970703125 0.37890625 0.5400390625 0.4453125 0.5517578125 0.4697265625 0.544921875 0.4775390625 0.533203125 0.4931640625 0.53125 0.5224609375 0.509765625


================================================
FILE: TumorDetection/train/labels/pituitary_449_jpg.rf.6d75373e9387e79e037bdac0438c0e6f.txt
================================================
3 0.5068359375 0.4921875 0.455078125 0.5263671875 0.447265625 0.5458984375 0.4375 0.5458984375 0.43359375 0.5673828125 0.4521484375 0.576171875 0.4736328125 0.607421875 0.521484375 0.6123046875 0.55078125 0.5888671875 0.560546875 0.5341796875 0.5361328125 0.498046875 0.5068359375 0.4921875


================================================
FILE: TumorDetection/train/labels/pituitary_44_jpg.rf.c48004d2e5932f2334c827524d658c8b.txt
================================================
3 0.5 0.5537109375 0.5078125 0.5029296875 0.49609375 0.4755859375 0.4716796875 0.447265625 0.4189453125 0.416015625 0.3896484375 0.412109375 0.3505859375 0.427734375 0.326171875 0.4736328125 0.3046875 0.4912109375 0.3046875 0.5439453125 0.3291015625 0.54296875 0.33203125 0.5498046875 0.291015625 0.5771484375 0.291015625 0.6396484375 0.30078125 0.6767578125 0.3154296875 0.6875 0.3935546875 0.70703125 0.431640625 0.7080078125 0.45703125 0.6865234375 0.45703125 0.5947265625 0.4755859375 0.583984375 0.5 0.5537109375


================================================
FILE: TumorDetection/train/labels/pituitary_465_jpg.rf.f3145501ee162dfcbecbafa0153ebefc.txt
================================================
3 0.5107421875 0.392578125 0.4755859375 0.400390625 0.466796875 0.4111328125 0.466796875 0.4345703125 0.48046875 0.4638671875 0.458984375 0.4931640625 0.48828125 0.4833984375 0.474609375 0.4794921875 0.4794921875 0.466796875 0.5341796875 0.46875 0.5419921875 0.4609375 0.55078125 0.4736328125 0.5419921875 0.48046875 0.490234375 0.4833984375 0.5244140625 0.490234375 0.5439453125 0.482421875 0.5625 0.4970703125 0.552734375 0.4638671875 0.5625 0.4345703125 0.5546875 0.4072265625 0.5107421875 0.392578125


================================================
FILE: TumorDetection/train/labels/pituitary_473_jpg.rf.e1948b14f933edfeade450f357db08f3.txt
================================================
3 0.552734375 0.5224609375 0.55078125 0.5068359375 0.525390625 0.4873046875 0.5048828125 0.447265625 0.4775390625 0.44140625 0.4453125 0.4677734375 0.4453125 0.5302734375 0.4794921875 0.5625 0.529296875 0.5634765625 0.537109375 0.5556640625 0.53125 0.5322265625 0.5400390625 0.521484375 0.552734375 0.5224609375


================================================
FILE: TumorDetection/train/labels/pituitary_482_jpg.rf.dd6b990733496ceda8f4fad4a9d438e7.txt
================================================
3 0.5546875 0.4736328125 0.5361328125 0.44140625 0.5048828125 0.431640625 0.4609375 0.4580078125 0.453125 0.4912109375 0.45703125 0.5068359375 0.4755859375 0.51953125 0.52734375 0.5244140625 0.5439453125 0.51171875 0.55859375 0.5107421875 0.552734375 0.5029296875 0.5546875 0.4736328125


================================================
FILE: TumorDetection/train/labels/pituitary_516_jpg.rf.3145996fbbb24667aac495d31785f94f.txt
================================================
3 0.44140625 0.3974609375 0.4443359375 0.40625 0.4814453125 0.3984375 0.568359375 0.4052734375 0.5556640625 0.375 0.4970703125 0.36328125 0.4638671875 0.365234375 0.44140625 0.3974609375


================================================
FILE: TumorDetection/train/labels/pituitary_519_jpg.rf.f98a7a057186388f5dabbce0af86eac0.txt
================================================
3 0.5302734375 0.548828125 0.5087890625 0.548828125 0.5009765625 0.55859375 0.4638671875 0.548828125 0.4501953125 0.5546875 0.44140625 0.5947265625 0.45703125 0.6240234375 0.4775390625 0.640625 0.5078125 0.6435546875 0.541015625 0.6298828125 0.556640625 0.5986328125 0.5546875 0.5673828125 0.5302734375 0.548828125


================================================
FILE: TumorDetection/train/labels/pituitary_51_jpg.rf.c86c7c9cb6f00d5d6a5ebf6374ff55a0.txt
================================================
3 0.3974609375 0.4296875 0.3505859375 0.423828125 0.32421875 0.4462890625 0.3125 0.5048828125 0.314453125 0.5654296875 0.3349609375 0.59375 0.36328125 0.5966796875 0.41796875 0.5693359375 0.435546875 0.4970703125 0.421875 0.4580078125 0.3974609375 0.4296875


================================================
FILE: TumorDetection/train/labels/pituitary_55_jpg.rf.4973b5b39b5831b2fae3112a0764d24c.txt
================================================
3 0.3896484375 0.455078125 0.3525390625 0.455078125 0.322265625 0.4736328125 0.310546875 0.5302734375 0.322265625 0.5556640625 0.3447265625 0.578125 0.390625 0.5849609375 0.43359375 0.5693359375 0.4375 0.5048828125 0.431640625 0.4912109375 0.3896484375 0.455078125


================================================
FILE: TumorDetection/train/labels/pituitary_566_jpg.rf.36dc9507f01dda158361fec1e1656203.txt
================================================
3 0.46484375 0.4130859375 0.466796875 0.4462890625 0.4755859375 0.453125 0.533203125 0.4560546875 0.533203125 0.4384765625 0.5029296875 0.40234375 0.4833984375 0.3984375 0.46484375 0.4130859375


================================================
FILE: TumorDetection/train/labels/pituitary_571_jpg.rf.9932cf26092b727e163954279e53f3f3.txt
================================================
3 0.564453125 0.6142578125 0.533203125 0.5849609375 0.55859375 0.5498046875 0.55078125 0.5244140625 0.5048828125 0.494140625 0.4755859375 0.49609375 0.4345703125 0.515625 0.416015625 0.5380859375 0.419921875 0.5634765625 0.44140625 0.5947265625 0.41796875 0.6142578125 0.431640625 0.6376953125 0.4619140625 0.6171875 0.4892578125 0.625 0.5205078125 0.615234375 0.552734375 0.6416015625 0.580078125 0.6845703125 0.583984375 0.6630859375 0.568359375 0.6318359375 0.552734375 0.6259765625 0.564453125 0.6142578125


================================================
FILE: TumorDetection/train/labels/pituitary_575_jpg.rf.797a6694b0f849b807d6bc0761462b06.txt
================================================
3 0.5146484375 0.3515625 0.4775390625 0.357421875 0.46484375 0.3720703125 0.45703125 0.4306640625 0.4716796875 0.44140625 0.517578125 0.4404296875 0.5380859375 0.43359375 0.544921875 0.4228515625 0.537109375 0.3720703125 0.5146484375 0.3515625


================================================
FILE: TumorDetection/train/labels/pituitary_588_jpg.rf.f2ced73df65f8bb544a762fb33e72f0e.txt
================================================
3 0.5517578125 0.59375 0.5048828125 0.583984375 0.4873046875 0.58984375 0.455078125 0.6259765625 0.458984375 0.6943359375 0.4853515625 0.705078125 0.5234375 0.7041015625 0.5380859375 0.69921875 0.580078125 0.6591796875 0.58203125 0.6416015625 0.5517578125 0.59375


================================================
FILE: TumorDetection/train/labels/pituitary_590_jpg.rf.e84608167ffa04939c2bb714fc8380a5.txt
================================================
3 0.591796875 0.3564453125 0.5654296875 0.326171875 0.4951171875 0.322265625 0.447265625 0.3662109375 0.447265625 0.4619140625 0.4697265625 0.486328125 0.5078125 0.4931640625 0.5615234375 0.482421875 0.60546875 0.4462890625 0.603515625 0.3837890625 0.591796875 0.3564453125


================================================
FILE: TumorDetection/train/labels/pituitary_596_jpg.rf.554a600b1f88405d132b160f1229ad36.txt
================================================
3 0.61328125 0.6005859375 0.5888671875 0.56640625 0.4873046875 0.5703125 0.4716796875 0.591796875 0.451171875 0.6005859375 0.44921875 0.6572265625 0.462890625 0.6943359375 0.4853515625 0.703125 0.51171875 0.7021484375 0.5810546875 0.681640625 0.591796875 0.6513671875 0.61328125 0.6455078125 0.61328125 0.6005859375


================================================
FILE: TumorDetection/train/labels/pituitary_612_jpg.rf.68bf84d673ece7bddabaef4ff4892665.txt
================================================
3 0.4931640625 0.296875 0.4638671875 0.287109375 0.4443359375 0.294921875 0.431640625 0.3115234375 0.42578125 0.3720703125 0.4482421875 0.396484375 0.4677734375 0.40625 0.486328125 0.4052734375 0.515625 0.3857421875 0.525390625 0.3603515625 0.5234375 0.3408203125 0.4931640625 0.296875


================================================
FILE: TumorDetection/train/labels/pituitary_613_jpg.rf.fb80ec77fc161e85db9485b9ca93520d.txt
================================================
3 0.634765625 0.6533203125 0.62109375 0.6142578125 0.6064453125 0.59765625 0.5771484375 0.583984375 0.5673828125 0.587890625 0.5146484375 0.54296875 0.5029296875 0.5234375 0.4912109375 0.5234375 0.47265625 0.5810546875 0.44921875 0.6005859375 0.443359375 0.6416015625 0.4873046875 0.69921875 0.5068359375 0.708984375 0.541015625 0.7119140625 0.5615234375 0.689453125 0.5830078125 0.6875 0.6298828125 0.6640625 0.634765625 0.6533203125


================================================
FILE: TumorDetection/train/labels/pituitary_632_jpg.rf.8fdf813e1806f69c947425bbbf6239dd.txt
================================================
3 0.587890625 0.5009765625 0.5537109375 0.462890625 0.5087890625 0.458984375 0.4482421875 0.46875 0.4296875 0.4990234375 0.43359375 0.5556640625 0.4736328125 0.607421875 0.525390625 0.6103515625 0.5458984375 0.603515625 0.5859375 0.5537109375 0.591796875 0.5400390625 0.587890625 0.5009765625


================================================
FILE: TumorDetection/train/labels/pituitary_633_jpg.rf.996b1eb1f3d572ca7f8638174a69f144.txt
================================================
3 0.5859375 0.5224609375 0.5478515625 0.48046875 0.4658203125 0.4765625 0.4501953125 0.482421875 0.431640625 0.5126953125 0.431640625 0.5556640625 0.4833984375 0.62109375 0.53125 0.6201171875 0.5546875 0.5986328125 0.560546875 0.5810546875 0.587890625 0.5595703125 0.5859375 0.5224609375


================================================
FILE: TumorDetection/train/labels/pituitary_635_jpg.rf.9aa64b0eb54322fe8965104ada04de79.txt
================================================
3 0.5146484375 0.34375 0.4951171875 0.3515625 0.4482421875 0.34375 0.443359375 0.3544921875 0.44921875 0.3759765625 0.435546875 0.3994140625 0.439453125 0.4208984375 0.4638671875 0.443359375 0.494140625 0.4462890625 0.533203125 0.4287109375 0.541015625 0.4091796875 0.5390625 0.3876953125 0.5146484375 0.34375


================================================
FILE: TumorDetection/train/labels/pituitary_636_jpg.rf.4adb9120661b9d4e1728ebe83b4cabc4.txt
================================================
3 0.4814453125 0.353515625 0.4580078125 0.36328125 0.443359375 0.3798828125 0.4375 0.4130859375 0.4619140625 0.4453125 0.49609375 0.4501953125 0.533203125 0.4267578125 0.54296875 0.3916015625 0.5244140625 0.36328125 0.4814453125 0.353515625


================================================
FILE: TumorDetection/train/labels/pituitary_653_jpg.rf.2f24641dd8e8005ef537e44e45e62962.txt
================================================
3 0.5166015625 0.833984375 0.5390625 0.8330078125 0.57421875 0.8095703125 0.5107421875 0.7890625 0.478515625 0.8134765625 0.5166015625 0.833984375
3 0.4794921875 0.6171875 0.4765625 0.6396484375 0.4951171875 0.65234375 0.509765625 0.6513671875 0.5498046875 0.646484375 0.546875 0.6220703125 0.5126953125 0.60546875 0.4794921875 0.6171875
3 0.7314453125 0.583984375 0.720703125 0.5966796875 0.720703125 0.6123046875 0.73046875 0.6455078125 0.744140625 0.6552734375 0.759765625 0.6435546875 0.759765625 0.6123046875 0.75390625 0.5927734375 0.7314453125 0.583984375


================================================
FILE: TumorDetection/train/labels/pituitary_687_jpg.rf.f0006d3aae44dc3214e24824c1266c6b.txt
================================================
3 0.572265625 0.4306640625 0.5341796875 0.408203125 0.5107421875 0.419921875 0.4697265625 0.416015625 0.4462890625 0.427734375 0.43359375 0.4736328125 0.443359375 0.5087890625 0.4775390625 0.55078125 0.529296875 0.5537109375 0.5546875 0.5341796875 0.576171875 0.4931640625 0.580078125 0.4541015625 0.572265625 0.4306640625


================================================
FILE: TumorDetection/train/labels/pituitary_694_jpg.rf.19db15b9effa9045595c060a8a99c38f.txt
================================================
3 0.5400390625 0.32421875 0.5126953125 0.31640625 0.4619140625 0.328125 0.435546875 0.3662109375 0.44921875 0.4169921875 0.4716796875 0.4375 0.498046875 0.4443359375 0.5439453125 0.431640625 0.5703125 0.3759765625 0.560546875 0.3427734375 0.5400390625 0.32421875


================================================
FILE: TumorDetection/train/labels/pituitary_696_jpg.rf.90761be4dcae375307c1effc13d35364.txt
================================================
3 0.4931640625 0.36328125 0.447265625 0.3935546875 0.4658203125 0.4140625 0.4716796875 0.41015625 0.482421875 0.4189453125 0.47265625 0.4287109375 0.4833984375 0.44140625 0.541015625 0.4423828125 0.556640625 0.4072265625 0.556640625 0.3916015625 0.5478515625 0.384765625 0.4931640625 0.36328125


================================================
FILE: TumorDetection/train/labels/pituitary_6_jpg.rf.2ca208e02910662c1dae7d012c5ae6c7.txt
================================================
3 0.53515625 0.5673828125 0.55078125 0.5458984375 0.544921875 0.5009765625 0.533203125 0.4990234375 0.5302734375 0.484375 0.5234375 0.4853515625 0.521484375 0.5009765625 0.5029296875 0.513671875 0.4892578125 0.515625 0.482421875 0.5087890625 0.4765625 0.4970703125 0.49609375 0.4873046875 0.484375 0.4833984375 0.49609375 0.4794921875 0.4912109375 0.470703125 0.4677734375 0.48046875 0.443359375 0.5068359375 0.4296875 0.5341796875 0.435546875 0.5478515625 0.392578125 0.5751953125 0.37890625 0.6005859375 0.373046875 0.6474609375 0.3837890625 0.65625 0.4228515625 0.650390625 0.4326171875 0.669921875 0.4912109375 0.685546875 0.51953125 0.6845703125 0.5673828125 0.658203125 0.59375 0.6142578125 0.5712890625 0.578125 0.53515625 0.5673828125
3 0.4296875 0.5322265625 0.4404296875 0.548828125 0.46484375 0.5498046875 0.478515625 0.5380859375 0.4580078125 0.517578125 0.4462890625 0.517578125 0.4296875 0.5322265625


================================================
FILE: TumorDetection/train/labels/pituitary_709_jpg.rf.d810bc61ee13a6dac8cc402cfdf485b5.txt
================================================
3 0.4814453125 0.373046875 0.458984375 0.4033203125 0.462890625 0.4345703125 0.4833984375 0.451171875 0.52734375 0.4521484375 0.5390625 0.4384765625 0.529296875 0.3935546875 0.5087890625 0.373046875 0.4814453125 0.373046875


================================================
FILE: TumorDetection/train/labels/pituitary_736_jpg.rf.a6597967353c35a003c6b083d31835e0.txt
================================================
3 0.5859375 0.4755859375 0.546875 0.4267578125 0.544921875 0.3974609375 0.5341796875 0.37890625 0.4658203125 0.3671875 0.447265625 0.3837890625 0.443359375 0.4130859375 0.44921875 0.4326171875 0.42578125 0.4677734375 0.4375 0.4794921875 0.4404296875 0.5078125 0.5068359375 0.541015625 0.51953125 0.5400390625 0.5263671875 0.53125 0.5498046875 0.52734375 0.5791015625 0.501953125 0.5859375 0.4755859375


================================================
FILE: TumorDetection/train/labels/pituitary_745_jpg.rf.db50f0c0c6d89535fd892b476f3bdc0e.txt
================================================
3 0.58984375 0.6005859375 0.5615234375 0.57421875 0.5009765625 0.57421875 0.4580078125 0.5859375 0.4365234375 0.59375 0.41796875 0.6201171875 0.42578125 0.6474609375 0.419921875 0.6767578125 0.4453125 0.7255859375 0.4794921875 0.751953125 0.505859375 0.7509765625 0.537109375 0.7275390625 0.564453125 0.6630859375 0.58984375 0.6455078125 0.58984375 0.6005859375


================================================
FILE: TumorDetection/train/labels/pituitary_747_jpg.rf.1cd94a1e17168f351aefbd2e1aecaaa4.txt
================================================
3 0.54296875 0.4599609375 0.5302734375 0.4453125 0.5009765625 0.4453125 0.478515625 0.4638671875 0.4697265625 0.49609375 0.443359375 0.5009765625 0.439453125 0.5146484375 0.4501953125 0.525390625 0.4697265625 0.537109375 0.484375 0.5361328125 0.55078125 0.5166015625 0.54296875 0.4931640625 0.533203125 0.4892578125 0.54296875 0.4599609375


================================================
FILE: TumorDetection/train/labels/pituitary_766_jpg.rf.bd99f9b94c920fe79998c62f45f9e47d.txt
================================================
3 0.4970703125 0.521484375 0.4619140625 0.498046875 0.4453125 0.5009765625 0.44921875 0.5283203125 0.4736328125 0.5546875 0.5166015625 0.55859375 0.5439453125 0.576171875 0.564453125 0.5771484375 0.5703125 0.5341796875 0.5595703125 0.5078125 0.5224609375 0.5078125 0.4970703125 0.521484375


================================================
FILE: TumorDetection/train/labels/pituitary_768_jpg.rf.e8a430da7c5b0950e7df9462e5e4de26.txt
================================================
3 0.4921875 0.5029296875 0.4794921875 0.501953125 0.4580078125 0.51171875 0.443359375 0.5322265625 0.439453125 0.5771484375 0.4599609375 0.599609375 0.498046875 0.6005859375 0.517578125 0.5849609375 0.521484375 0.5537109375 0.494140625 0.5263671875 0.4921875 0.5029296875


================================================
FILE: TumorDetection/train/labels/pituitary_784_jpg.rf.35fbd8d5e826518335d6983657dafb52.txt
================================================
3 0.5556640625 0.4609375 0.5087890625 0.42578125 0.4658203125 0.41796875 0.4453125 0.4306640625 0.4375 0.4814453125 0.423828125 0.4931640625 0.423828125 0.5166015625 0.4482421875 0.544921875 0.494140625 0.5537109375 0.5693359375 0.53515625 0.578125 0.5244140625 0.580078125 0.4912109375 0.5556640625 0.4609375


================================================
FILE: TumorDetection/train/labels/pituitary_796_jpg.rf.9c5dc3279b211487c125bfb7cf39a3b8.txt
================================================
3 0.58984375 0.5419921875 0.5791015625 0.521484375 0.5087890625 0.517578125 0.4794921875 0.5234375 0.4599609375 0.537109375 0.4296875 0.5380859375 0.4296875 0.5732421875 0.439453125 0.5869140625 0.4482421875 0.59375 0.4892578125 0.587890625 0.49609375 0.5947265625 0.494140625 0.6083984375 0.5078125 0.6181640625 0.517578125 0.6123046875 0.5283203125 0.5859375 0.59375 0.5849609375 0.58984375 0.5419921875


================================================
FILE: TumorDetection/train/labels/pituitary_811_jpg.rf.1615ed84247b9d492b270de36e664e8e.txt
================================================
3 0.521484375 0.5751953125 0.517578125 0.5556640625 0.4931640625 0.544921875 0.4384765625 0.580078125 0.41796875 0.5830078125 0.423828125 0.5986328125 0.408203125 0.6162109375 0.408203125 0.6416015625 0.4140625 0.6669921875 0.4267578125 0.681640625 0.4814453125 0.701171875 0.51171875 0.7001953125 0.541015625 0.6767578125 0.552734375 0.6259765625 0.548828125 0.6025390625 0.521484375 0.5751953125


================================================
FILE: TumorDetection/train/labels/pituitary_812_jpg.rf.8049b42fd2d7555107945f85d74f0cb0.txt
================================================
3 0.52734375 0.5810546875 0.5087890625 0.5234375 0.4951171875 0.5234375 0.4921875 0.5361328125 0.4345703125 0.5859375 0.3994140625 0.59375 0.380859375 0.6123046875 0.365234375 0.6533203125 0.4013671875 0.6875 0.4365234375 0.689453125 0.4599609375 0.712890625 0.4765625 0.7119140625 0.5126953125 0.69921875 0.5546875 0.6435546875 0.55078125 0.5986328125 0.52734375 0.5810546875


================================================
FILE: TumorDetection/train/labels/pituitary_817_jpg.rf.4bc6eec52131d9d6e2dac873123dd565.txt
================================================
3 0.62890625 0.4482421875 0.6142578125 0.43359375 0.5888671875 0.42578125 0.5537109375 0.427734375 0.529296875 0.4658203125 0.537109375 0.5517578125 0.5498046875 0.583984375 0.60546875 0.6005859375 0.63671875 0.5771484375 0.646484375 0.5419921875 0.642578125 0.4814453125 0.62890625 0.4482421875


================================================
FILE: TumorDetection/train/labels/pituitary_831_jpg.rf.591ba244eb69a0d6bd26de903a4df1bf.txt
================================================
3 0.6240234375 0.494140625 0.5830078125 0.482421875 0.53515625 0.5009765625 0.529296875 0.5380859375 0.533203125 0.6044921875 0.5478515625 0.642578125 0.609375 0.6552734375 0.6240234375 0.650390625 0.650390625 0.6162109375 0.654296875 0.5322265625 0.6240234375 0.494140625


================================================
FILE: TumorDetection/train/labels/pituitary_838_jpg.rf.82b13975e57d46c90838e2b4e6aa77ef.txt
================================================
3 0.6103515625 0.509765625 0.5634765625 0.51171875 0.548828125 0.5283203125 0.548828125 0.5693359375 0.5615234375 0.5859375 0.5810546875 0.595703125 0.599609375 0.5947265625 0.6123046875 0.59375 0.623046875 0.5791015625 0.630859375 0.5283203125 0.6103515625 0.509765625


================================================
FILE: TumorDetection/train/labels/pituitary_860_jpg.rf.91c08cf0b866472922fc62d1bf8e086b.txt
================================================
3 0.5126953125 0.48046875 0.484375 0.5009765625 0.474609375 0.5263671875 0.4892578125 0.533203125 0.5146484375 0.56640625 0.541015625 0.5693359375 0.568359375 0.5537109375 0.580078125 0.5146484375 0.5517578125 0.490234375 0.5126953125 0.48046875


================================================
FILE: TumorDetection/train/labels/pituitary_862_jpg.rf.a469356b04ff392e126c0d56d8eb52e0.txt
================================================
3 0.5205078125 0.48046875 0.505859375 0.4912109375 0.50390625 0.5302734375 0.5302734375 0.560546875 0.55859375 0.5673828125 0.57421875 0.5517578125 0.583984375 0.5185546875 0.5595703125 0.490234375 0.5205078125 0.48046875


================================================
FILE: TumorDetection/train/labels/pituitary_883_jpg.rf.41b151d1be030e3197206cc341a8f3b8.txt
================================================
3 0.5693359375 0.3359375 0.5048828125 0.310546875 0.45703125 0.3349609375 0.443359375 0.3642578125 0.443359375 0.3876953125 0.4697265625 0.423828125 0.525390625 0.4306640625 0.5419921875 0.42578125 0.560546875 0.4072265625 0.580078125 0.3525390625 0.5693359375 0.3359375


================================================
FILE: TumorDetection/train/labels/pituitary_884_jpg.rf.a62e2da57ac8687dad3815d5588d1f2e.txt
================================================
3 0.6103515625 0.4375 0.5654296875 0.427734375 0.52734375 0.4482421875 0.515625 0.4736328125 0.5234375 0.5263671875 0.546875 0.5322265625 0.5908203125 0.521484375 0.6171875 0.4951171875 0.623046875 0.4658203125 0.6103515625 0.4375


================================================
FILE: TumorDetection/train/labels/pituitary_890_jpg.rf.400ed548e401014b561dd674f5b793d7.txt
================================================
3 0.5361328125 0.484375 0.5068359375 0.490234375 0.486328125 0.5244140625 0.49609375 0.5634765625 0.52734375 0.5830078125 0.556640625 0.5673828125 0.568359375 0.5302734375 0.5625 0.5048828125 0.5361328125 0.484375


================================================
FILE: TumorDetection/train/labels/pituitary_921_jpg.rf.4f9bc7ce14afa06a9f462f4b1e052962.txt
================================================
3 0.4501953125 0.361328125 0.44140625 0.3974609375 0.451171875 0.4443359375 0.4697265625 0.462890625 0.50390625 0.4658203125 0.529296875 0.4482421875 0.53125 0.3916015625 0.5107421875 0.365234375 0.4501953125 0.361328125


================================================
FILE: TumorDetection/train/labels/pituitary_930_jpg.rf.bab1001bd2e968752716a86bae781090.txt
================================================
3 0.56640625 0.5595703125 0.5703125 0.5341796875 0.556640625 0.5107421875 0.5341796875 0.494140625 0.4638671875 0.486328125 0.421875 0.5146484375 0.41796875 0.5400390625 0.4501953125 0.603515625 0.494140625 0.6142578125 0.5185546875 0.61328125 0.537109375 0.5986328125 0.546875 0.5771484375 0.56640625 0.5595703125


================================================
FILE: TumorDetection/train/labels/pituitary_932_jpg.rf.aa24e9a8430e9d1c448c74ae8e37e70d.txt
================================================
3 0.4853515625 0.408203125 0.466796875 0.4267578125 0.484375 0.4638671875 0.498046875 0.4716796875 0.52734375 0.4345703125 0.525390625 0.4189453125 0.5087890625 0.40625 0.4853515625 0.408203125


================================================
FILE: TumorDetection/train/labels/pituitary_940_jpg.rf.54a9faf38d85dc767cf4a9ce0af0d5f0.txt
================================================
3 0.5439453125 0.51171875 0.4873046875 0.513671875 0.4423828125 0.505859375 0.404296875 0.5458984375 0.40625 0.5927734375 0.431640625 0.6416015625 0.4560546875 0.6640625 0.505859375 0.6708984375 0.576171875 0.6025390625 0.5859375 0.5673828125 0.58203125 0.5419921875 0.5439453125 0.51171875


================================================
FILE: TumorDetection/train/labels/pituitary_961_jpg.rf.1563674e159c7cb834396022137202dc.txt
================================================
3 0.4755859375 0.349609375 0.443359375 0.3662109375 0.4375 0.3994140625 0.4521484375 0.419921875 0.4736328125 0.427734375 0.521484375 0.4267578125 0.5390625 0.3857421875 0.5263671875 0.357421875 0.5087890625 0.349609375 0.4755859375 0.349609375


================================================
FILE: TumorDetection/train/labels/pituitary_976_jpg.rf.c2692a31bc999aeaeec2fe9e844e173a.txt
================================================
3 0.4990234375 0.37890625 0.4794921875 0.388671875 0.470703125 0.4052734375 0.474609375 0.4462890625 0.4951171875 0.45703125 0.546875 0.4619140625 0.564453125 0.4443359375 0.556640625 0.3935546875 0.5263671875 0.37890625 0.4990234375 0.37890625


================================================
FILE: TumorDetection/train/labels/pituitary_997_jpg.rf.e360b7003605a483342546daed1d7f8e.txt
================================================
3 0.4931640625 0.37109375 0.4755859375 0.37890625 0.45703125 0.4052734375 0.455078125 0.4287109375 0.4755859375 0.4609375 0.50390625 0.4638671875 0.53125 0.4443359375 0.544921875 0.4111328125 0.5302734375 0.380859375 0.4931640625 0.37109375


================================================
FILE: TumorDetection/valid/labels.cache
================================================
[Non-text file]



================================================
FILE: TumorDetection/valid/labels/glioma_1022_jpg.rf.ab6956aa8c8a25f994539c5cf6227064.txt
================================================
1 0.5771484375 0.40625 0.5419921875 0.388671875 0.5126953125 0.384765625 0.48828125 0.4345703125 0.484375 0.4892578125 0.564453125 0.5107421875 0.5830078125 0.505859375 0.619140625 0.4775390625 0.619140625 0.4482421875 0.5771484375 0.40625


================================================
FILE: TumorDetection/valid/labels/glioma_104_jpg.rf.f5deabc016a8b3a913888b09baebacee.txt
================================================
1 0.5712890625 0.44921875 0.5400390625 0.44140625 0.517578125 0.4052734375 0.5234375 0.3798828125 0.5146484375 0.376953125 0.5 0.3876953125 0.5 0.4189453125 0.5361328125 0.451171875 0.5869140625 0.46875 0.62109375 0.4677734375 0.625 0.4638671875 0.5966796875 0.451171875 0.5986328125 0.4453125 0.5712890625 0.44921875


================================================
FILE: TumorDetection/valid/labels/glioma_1088_jpg.rf.5542c8b3dc2add56cd7303d7007e3ae8.txt
================================================
1 0.6337890625 0.44140625 0.6103515625 0.42578125 0.5478515625 0.408203125 0.5166015625 0.41796875 0.482421875 0.4482421875 0.490234375 0.5029296875 0.5087890625 0.53125 0.626953125 0.5634765625 0.65625 0.5498046875 0.66796875 0.5166015625 0.654296875 0.4697265625 0.6337890625 0.44140625


================================================
FILE: TumorDetection/valid/labels/glioma_1109_jpg.rf.710d66962bf0db65050c34e750be6e7a.txt
================================================
1 0.7119140625 0.306640625 0.6962890625 0.302734375 0.6728515625 0.306640625 0.6328125 0.3447265625 0.6376953125 0.37109375 0.6728515625 0.39453125 0.6875 0.3955078125 0.7216796875 0.37890625 0.736328125 0.3544921875 0.7265625 0.3232421875 0.7119140625 0.306640625


================================================
FILE: TumorDetection/valid/labels/glioma_1164_jpg.rf.4f2cfa1dc0e93548eeef3f9c30e3b7ee.txt
================================================
1 0.759765625 0.4638671875 0.7412109375 0.44921875 0.7021484375 0.453125 0.6953125 0.4365234375 0.671875 0.4208984375 0.669921875 0.3974609375 0.6494140625 0.376953125 0.6376953125 0.375 0.59765625 0.4365234375 0.59375 0.4755859375 0.6591796875 0.533203125 0.703125 0.5400390625 0.73046875 0.5244140625 0.732421875 0.5048828125 0.7578125 0.4775390625 0.759765625 0.4638671875


================================================
FILE: TumorDetection/valid/labels/glioma_1226_jpg.rf.b459e0d24ecd906c121e9be1b88907c4.txt
================================================
1 0.3896484375 0.482421875 0.3994140625 0.47265625 0.4208984375 0.47265625 0.4287109375 0.482421875 0.4296875 0.4716796875 0.4111328125 0.458984375 0.3271484375 0.45703125 0.3125 0.5048828125 0.328125 0.5146484375 0.32421875 0.4970703125 0.333984375 0.4873046875 0.3310546875 0.48046875 0.3896484375 0.482421875


================================================
FILE: TumorDetection/valid/labels/glioma_1238_jpg.rf.e7fec135cbd5bafba3674a18b0b85818.txt
================================================
1 0.5302734375 0.162109375 0.4736328125 0.16796875 0.46875 0.1845703125 0.498046875 0.2431640625 0.5302734375 0.263671875 0.548828125 0.2626953125 0.564453125 0.2509765625 0.564453125 0.1923828125 0.5546875 0.1728515625 0.5302734375 0.162109375
1 0.4501953125 0.298828125 0.4306640625 0.291015625 0.423828125 0.2978515625 0.4296875 0.3056640625 0.42578125 0.3193359375 0.44921875 0.3251953125 0.4443359375 0.33203125 0.4296875 0.3291015625 0.4453125 0.3603515625 0.4736328125 0.376953125 0.4794921875 0.388671875 0.50390625 0.3916015625 0.515625 0.3798828125 0.50390625 0.3154296875 0.4775390625 0.296875 0.4501953125 0.298828125
1 0.4208984375 0.3984375 0.3798828125 0.34375 0.3447265625 0.33984375 0.3154296875 0.34765625 0.3125 0.3623046875 0.330078125 0.3916015625 0.333984375 0.4228515625 0.349609375 0.4384765625 0.34765625 0.4638671875 0.3701171875 0.484375 0.390625 0.4873046875 0.4384765625 0.4765625 0.451171875 0.4638671875 0.44140625 0.4033203125 0.4208984375 0.3984375


================================================
FILE: TumorDetection/valid/labels/glioma_1254_jpg.rf.809e3b791a236a04a1445e2c5f7d979e.txt
================================================
1 0.4833984375 0.234375 0.4638671875 0.236328125 0.443359375 0.2509765625 0.443359375 0.2900390625 0.4736328125 0.32421875 0.49609375 0.3271484375 0.533203125 0.3056640625 0.53515625 0.2724609375 0.5126953125 0.244140625 0.4833984375 0.234375


================================================
FILE: TumorDetection/valid/labels/glioma_1255_jpg.rf.ede8d4e550bf157ed2c9faf53dbeef4c.txt
================================================
1 0.55859375 0.2978515625 0.576171875 0.2705078125 0.57421875 0.2548828125 0.5615234375 0.240234375 0.5302734375 0.2265625 0.4677734375 0.23046875 0.451171875 0.2568359375 0.451171875 0.2705078125 0.474609375 0.3271484375 0.4931640625 0.3359375 0.5146484375 0.322265625 0.5205078125 0.3359375 0.5390625 0.3388671875 0.556640625 0.3271484375 0.552734375 0.3115234375 0.5390625 0.3017578125 0.55859375 0.2978515625


================================================
FILE: TumorDetection/valid/labels/glioma_1265_jpg.rf.0ceada97135f316a54f843aed7d1703e.txt
================================================
1 0.546875 0.4521484375 0.4873046875 0.41015625 0.4130859375 0.39453125 0.3876953125 0.3984375 0.3671875 0.4287109375 0.3828125 0.4951171875 0.4560546875 0.53515625 0.5078125 0.5439453125 0.5439453125 0.537109375 0.560546875 0.5185546875 0.560546875 0.4794921875 0.546875 0.4521484375


================================================
FILE: TumorDetection/valid/labels/glioma_1275_jpg.rf.b57270922a310c95831b5373c35f7e1f.txt
================================================
1 0.6611328125 0.599609375 0.638671875 0.6162109375 0.63671875 0.6416015625 0.6494140625 0.65625 0.669921875 0.6591796875 0.68359375 0.6474609375 0.69140625 0.6220703125 0.6748046875 0.603515625 0.6611328125 0.599609375


================================================
FILE: TumorDetection/valid/labels/glioma_12_jpg.rf.f146c6663e2614eba9da724a1c495acd.txt
================================================
1 0.69140625 0.3525390625 0.6640625 0.3310546875 0.658203125 0.3115234375 0.6416015625 0.30859375 0.5791015625 0.3359375 0.552734375 0.3681640625 0.548828125 0.3994140625 0.55859375 0.4345703125 0.5732421875 0.455078125 0.5927734375 0.46484375 0.6298828125 0.466796875 0.6474609375 0.48828125 0.6640625 0.4912109375 0.6689453125 0.478515625 0.712890625 0.4619140625 0.71484375 0.3955078125 0.69140625 0.3525390625


================================================
FILE: TumorDetection/valid/labels/glioma_1305_jpg.rf.9e8e8139db1dc7b685a00a340c0dcafd.txt
================================================
1 0.5341796875 0.70703125 0.51171875 0.6787109375 0.515625 0.6064453125 0.5048828125 0.59375 0.5 0.6201171875 0.4951171875 0.611328125 0.4921875 0.6162109375 0.4921875 0.6318359375 0.501953125 0.6279296875 0.49609375 0.6611328125 0.501953125 0.7177734375 0.564453125 0.7333984375 0.62890625 0.7119140625 0.6181640625 0.70703125 0.5595703125 0.716796875 0.5341796875 0.70703125


================================================
FILE: TumorDetection/valid/labels/glioma_154_jpg.rf.d6b8dbf3c9061876f1ee6dbbe6113664.txt
================================================
1 0.5048828125 0.390625 0.486328125 0.4033203125 0.46484375 0.4619140625 0.4677734375 0.470703125 0.5205078125 0.470703125 0.587890625 0.4833984375 0.587890625 0.4033203125 0.5771484375 0.384765625 0.5419921875 0.396484375 0.5048828125 0.390625


================================================
FILE: TumorDetection/valid/labels/glioma_167_jpg.rf.da25e4d04a053b942c6c5211a14397b3.txt
================================================
1 0.6005859375 0.380859375 0.546875 0.4091796875 0.55078125 0.4267578125 0.5791015625 0.4453125 0.658203125 0.4482421875 0.66796875 0.4345703125 0.666015625 0.4130859375 0.6201171875 0.380859375 0.6005859375 0.380859375


================================================
FILE: TumorDetection/valid/labels/glioma_201_jpg.rf.a847f9af4f209d6323a0c4e4f07d63f1.txt
================================================
1 0.484375 0.7666015625 0.50390625 0.7724609375 0.5234375 0.7568359375 0.5126953125 0.73828125 0.482421875 0.7509765625 0.484375 0.7666015625


================================================
FILE: TumorDetection/valid/labels/glioma_217_jpg.rf.8f0465bb2b7c3b57b0df37ff73a0259c.txt
================================================
1 0.4638671875 0.34375 0.443359375 0.3720703125 0.451171875 0.4404296875 0.4658203125 0.453125 0.509765625 0.4580078125 0.525390625 0.4462890625 0.517578125 0.4365234375 0.5 0.3466796875 0.4638671875 0.34375


================================================
FILE: TumorDetection/valid/labels/glioma_334_jpg.rf.08fe123f3cb4647ece300fc6aa648214.txt
================================================
1 0.4580078125 0.5234375 0.4423828125 0.5234375 0.400390625 0.5576171875 0.41015625 0.5849609375 0.4287109375 0.6015625 0.44140625 0.6025390625 0.4853515625 0.591796875 0.494140625 0.5791015625 0.48828125 0.5517578125 0.4580078125 0.5234375


================================================
FILE: TumorDetection/valid/labels/glioma_383_jpg.rf.c56f80ec1efa67d854350344b04f391d.txt
================================================
1 0.4931640625 0.578125 0.4873046875 0.572265625 0.484375 0.5849609375 0.4736328125 0.587890625 0.462890625 0.5751953125 0.46484375 0.5517578125 0.4560546875 0.546875 0.462890625 0.5908203125 0.455078125 0.6064453125 0.4677734375 0.62109375 0.48046875 0.6220703125 0.49609375 0.6103515625 0.505859375 0.5830078125 0.5009765625 0.57421875 0.4931640625 0.578125
1 0.4287109375 0.560546875 0.4150390625 0.56640625 0.3896484375 0.5625 0.375 0.5712890625 0.369140625 0.6240234375 0.37109375 0.6630859375 0.37890625 0.6748046875 0.3984375 0.6162109375 0.396484375 0.6005859375 0.431640625 0.5673828125 0.4287109375 0.560546875


================================================
FILE: TumorDetection/valid/labels/glioma_40_jpg.rf.a578edbb1cfe59a55b8377cdf4e46f16.txt
================================================
1 0.3544921875 0.40234375 0.34375 0.4326171875 0.380859375 0.4501953125 0.39453125 0.4404296875 0.40234375 0.4072265625 0.3857421875 0.3984375 0.3544921875 0.40234375


================================================
FILE: TumorDetection/valid/labels/glioma_428_jpg.rf.2d27302e3d74586debc5ab63e605671f.txt
================================================
1 0.271484375 0.3291015625 0.28125 0.3056640625 0.2607421875 0.2890625 0.2421875 0.3173828125 0.236328125 0.3759765625 0.2470703125 0.388671875 0.275390625 0.3916015625 0.3046875 0.3642578125 0.30859375 0.3466796875 0.2958984375 0.32421875 0.271484375 0.3291015625


================================================
FILE: TumorDetection/valid/labels/glioma_477_jpg.rf.fac2810f0b518f328de3369818336daf.txt
================================================
1 0.5478515625 0.560546875 0.546875 0.5426897328125 0.5205078125 0.564453125 0.5119977671875 0.5594308031249999 0.4423828125 0.556640625 0.421875 0.5654296875 0.4306640625 0.56640625 0.435546875 0.5771484375 0.4326171875 0.603515625 0.4638671875 0.623046875 0.5078125 0.6259765625 0.5361328125 0.60546875 0.5673828125 0.609375 0.58203125 0.5947265625 0.5478515625 0.560546875


================================================
FILE: TumorDetection/valid/labels/glioma_545_jpg.rf.1b2c558abef09235677541b8709a12b2.txt
================================================
1 0.634765625 0.2626953125 0.625 0.2529296875 0.6328125 0.2470703125 0.619140625 0.2314453125 0.6044921875 0.224609375 0.5927734375 0.228515625 0.5771484375 0.21484375 0.5302734375 0.203125 0.4970703125 0.2109375 0.48828125 0.2412109375 0.494140625 0.2841796875 0.5078125 0.3134765625 0.5263671875 0.322265625 0.5810546875 0.32421875 0.615234375 0.3388671875 0.63671875 0.3173828125 0.63671875 0.3076171875 0.64453125 0.3076171875 0.634765625 0.2626953125


================================================
FILE: TumorDetection/valid/labels/glioma_620_jpg.rf.aad32c4d1721799b3ad1149352cd3af5.txt
================================================
1 0.595703125 0.3623046875 0.595703125 0.3388671875 0.5869140625 0.33203125 0.5712890625 0.330078125 0.5185546875 0.369140625 0.4951171875 0.373046875 0.47265625 0.4052734375 0.478515625 0.4560546875 0.52734375 0.4619140625 0.53515625 0.4287109375 0.556640625 0.4150390625 0.580078125 0.3740234375 0.595703125 0.3623046875


================================================
FILE: TumorDetection/valid/labels/glioma_639_jpg.rf.b9c17f063e393ff2952e662a4af07ee2.txt
================================================
1 0.4404296875 0.4140625 0.41796875 0.4384765625 0.42578125 0.4853515625 0.4326171875 0.494140625 0.4453125 0.4931640625 0.47265625 0.4697265625 0.470703125 0.4326171875 0.4521484375 0.4140625 0.4404296875 0.4140625


================================================
FILE: TumorDetection/valid/labels/glioma_73_jpg.rf.c59c51554d141873ad71c7482447e854.txt
================================================
1 0.69921875 0.5556640625 0.68359375 0.5263671875 0.681640625 0.5048828125 0.6552734375 0.48828125 0.5771484375 0.513671875 0.552734375 0.5380859375 0.53515625 0.5732421875 0.57421875 0.6572265625 0.564453125 0.6630859375 0.57421875 0.6884765625 0.6064453125 0.736328125 0.62890625 0.7392578125 0.66796875 0.7216796875 0.73046875 0.6103515625 0.7265625 0.5771484375 0.69921875 0.5556640625


================================================
FILE: TumorDetection/valid/labels/glioma_76_jpg.rf.525d4172ac7731bdb68a73504014ffcc.txt
================================================
1 0.326171875 0.2666015625 0.3359375 0.2880859375 0.35546875 0.2978515625 0.373046875 0.2763671875 0.3623046875 0.2578125 0.3466796875 0.251953125 0.326171875 0.2666015625
1 0.556640625 0.5771484375 0.5322265625 0.552734375 0.470703125 0.5458984375 0.4921875 0.4541015625 0.48828125 0.4384765625 0.4697265625 0.423828125 0.455078125 0.4404296875 0.46875 0.4501953125 0.46875 0.4599609375 0.4453125 0.5166015625 0.427734375 0.5380859375 0.4375 0.5634765625 0.408203125 0.6064453125 0.40234375 0.6630859375 0.4248046875 0.716796875 0.478515625 0.7197265625 0.5732421875 0.71484375 0.591796875 0.6787109375 0.591796875 0.6552734375 0.556640625 0.5771484375


================================================
FILE: TumorDetection/valid/labels/glioma_937_jpg.rf.ae7c14cefc5df10cfd7a4a4c34ddfdf4.txt
================================================
1 0.4814453125 0.310546875 0.4638671875 0.302734375 0.431640625 0.3271484375 0.421875 0.3623046875 0.4423828125 0.3828125 0.48046875 0.3876953125 0.505859375 0.3701171875 0.515625 0.3369140625 0.5029296875 0.306640625 0.4814453125 0.310546875


================================================
FILE: TumorDetection/valid/labels/glioma_948_jpg.rf.f28cc5f42f837b7e005ffb2c740b8160.txt
================================================
1 0.4794921875 0.40625 0.462890625 0.4169921875 0.45703125 0.4306640625 0.4697265625 0.458984375 0.490234375 0.4619140625 0.5244140625 0.447265625 0.52734375 0.4365234375 0.5048828125 0.412109375 0.4794921875 0.40625
1 0.5537109375 0.37109375 0.59765625 0.4130859375 0.6015625 0.4326171875 0.591796875 0.4599609375 0.59765625 0.4931640625 0.6064453125 0.5078125 0.62890625 0.5185546875 0.623046875 0.4521484375 0.609375 0.4072265625 0.5771484375 0.375 0.5537109375 0.37109375


================================================
FILE: TumorDetection/valid/labels/glioma_964_jpg.rf.fc8440cfc0c94c2d452edebb7f71f0c5.txt
================================================
1 0.318359375 0.3837890625 0.318359375 0.4013671875 0.36328125 0.4365234375 0.37890625 0.4189453125 0.34765625 0.4013671875 0.3310546875 0.3671875 0.318359375 0.3837890625
1 0.4541015625 0.373046875 0.4208984375 0.384765625 0.380859375 0.4169921875 0.3916015625 0.41796875 0.4267578125 0.396484375 0.4697265625 0.390625 0.4990234375 0.39453125 0.5185546875 0.404296875 0.5341796875 0.42578125 0.546875 0.4267578125 0.560546875 0.4130859375 0.5283203125 0.384765625 0.4951171875 0.373046875 0.4541015625 0.373046875


================================================
FILE: TumorDetection/valid/labels/meningioma_1018_jpg.rf.31a71a9999537db1e0651387f1c2d102.txt
================================================
2 0.404296875 0.1552734375 0.3896484375 0.14453125 0.3544921875 0.1484375 0.318359375 0.1630859375 0.322265625 0.2138671875 0.3466796875 0.228515625 0.369140625 0.2275390625 0.3916015625 0.21875 0.40625 0.1943359375 0.41015625 0.1708984375 0.404296875 0.1552734375


================================================
FILE: TumorDetection/valid/labels/meningioma_1025_jpg.rf.642e70889054d7e7f3048e46948630a1.txt
================================================
2 0.73828125 0.3310546875 0.6611328125 0.26171875 0.630859375 0.3056640625 0.634765625 0.3447265625 0.6708984375 0.37109375 0.7080078125 0.365234375 0.73828125 0.3310546875


================================================
FILE: TumorDetection/valid/labels/meningioma_1044_jpg.rf.312ef7316cddb11bd721766f1712eced.txt
================================================
2 0.6708984375 0.68359375 0.697265625 0.6435546875 0.716796875 0.5947265625 0.703125 0.5498046875 0.6953125 0.5478515625 0.6943359375 0.572265625 0.6748046875 0.5703125 0.6572265625 0.578125 0.623046875 0.6103515625 0.619140625 0.6279296875 0.626953125 0.6630859375 0.6552734375 0.685546875 0.6708984375 0.68359375


================================================
FILE: TumorDetection/valid/labels/meningioma_108_jpg.rf.4b7febff023e20c52b0bda99f20322aa.txt
================================================
2 0.45703125 0.20379132812499998 0.45703125 0.23388132343749998 0.4658203109375 0.2516617734375 0.494140625 0.25576495625 0.505859375 0.201055875 0.4970703109375 0.1805399703125 0.4755859375 0.17506906093750002 0.45703125 0.20379132812499998


================================================
FILE: TumorDetection/valid/labels/meningioma_1123_jpg.rf.51f067df444ea84a088907c87805bdbe.txt
================================================
2 0.5986328125 0.265625 0.5341796875 0.28125 0.517578125 0.2998046875 0.517578125 0.3154296875 0.537109375 0.3564453125 0.5751953125 0.392578125 0.603515625 0.3955078125 0.634765625 0.3623046875 0.634765625 0.3095703125 0.625 0.2861328125 0.5986328125 0.265625


================================================
FILE: TumorDetection/valid/labels/meningioma_1125_jpg.rf.a8bfa701cc1fe4ce49a9e261ca79d4dd.txt
================================================
2 0.671875 0.3408203125 0.65234375 0.3291015625 0.634765625 0.3056640625 0.626953125 0.2763671875 0.6005859375 0.251953125 0.5830078125 0.26953125 0.5595703125 0.2578125 0.5283203125 0.265625 0.505859375 0.2900390625 0.505859375 0.3740234375 0.5439453125 0.400390625 0.6171875 0.4072265625 0.6474609375 0.3984375 0.6669921875 0.3671875 0.677734375 0.3701171875 0.671875 0.3408203125


================================================
FILE: TumorDetection/valid/labels/meningioma_115_jpg.rf.98156ea3bf9d25bbe5bc18bc6cc483b8.txt
================================================
2 0.4560546875 0.501953125 0.4716796875 0.5078125 0.5126953125 0.498046875 0.5380859375 0.501953125 0.552734375 0.4912109375 0.5546875 0.4599609375 0.544921875 0.4228515625 0.5244140625 0.400390625 0.4990234375 0.390625 0.4697265625 0.392578125 0.427734375 0.4169921875 0.421875 0.4384765625 0.4375 0.4599609375 0.443359375 0.4892578125 0.4560546875 0.501953125


================================================
FILE: TumorDetection/valid/labels/meningioma_1173_jpg.rf.b612f4d2f294a442496a0c61719c2c4b.txt
================================================
2 0.4443359375 0.279296875 0.4072265625 0.28125 0.3935546875 0.2890625 0.37109375 0.3154296875 0.373046875 0.3447265625 0.4052734375 0.373046875 0.4609375 0.3779296875 0.478515625 0.3642578125 0.486328125 0.3310546875 0.4765625 0.2978515625 0.4443359375 0.279296875


================================================
FILE: TumorDetection/valid/labels/meningioma_1184_jpg.rf.788f4bed9003550b08637f1f22bff24c.txt
================================================
2 0.57421875 0.5087890625 0.5341796875 0.484375 0.4794921875 0.5 0.4375 0.5361328125 0.419921875 0.5654296875 0.4365234375 0.611328125 0.4599609375 0.615234375 0.5078125 0.6435546875 0.5234375 0.6123046875 0.54296875 0.5361328125 0.578125 0.5302734375 0.57421875 0.5087890625


================================================
FILE: TumorDetection/valid/labels/meningioma_1189_jpg.rf.b46edce8b68e411674c47b753789fff6.txt
================================================
2 0.6435546875 0.298828125 0.5537109375 0.283203125 0.5166015625 0.310546875 0.498046875 0.3369140625 0.494140625 0.3662109375 0.50390625 0.3935546875 0.501953125 0.4248046875 0.5361328125 0.447265625 0.5751953125 0.443359375 0.607421875 0.4541015625 0.6357421875 0.443359375 0.677734375 0.4052734375 0.666015625 0.3251953125 0.6435546875 0.298828125


================================================
FILE: TumorDetection/valid/labels/meningioma_1196_jpg.rf.1ea0c6622f1af2932f65b9ae6d07b1f4.txt
================================================
2 0.3759765625 0.544921875 0.3037109375 0.560546875 0.2783203125 0.537109375 0.2412109375 0.541015625 0.23046875 0.5615234375 0.232421875 0.5888671875 0.216796875 0.6162109375 0.2265625 0.6650390625 0.3017578125 0.73046875 0.3671875 0.7392578125 0.3857421875 0.732421875 0.4453125 0.6806640625 0.4375 0.5927734375 0.3759765625 0.544921875
2 0.5712890625 0.427734375 0.5869140625 0.416015625 0.6162109375 0.416015625 0.6572265625 0.458984375 0.6728515625 0.458984375 0.697265625 0.4365234375 0.703125 0.3916015625 0.6748046875 0.349609375 0.6259765625 0.326171875 0.5556640625 0.310546875 0.4970703125 0.3125 0.4326171875 0.34765625 0.38671875 0.3974609375 0.38671875 0.4130859375 0.40234375 0.4443359375 0.4228515625 0.462890625 0.4375 0.4658203125 0.4404296875 0.447265625 0.5087890625 0.4140625 0.5498046875 0.4140625 0.5712890625 0.427734375


================================================
FILE: TumorDetection/valid/labels/meningioma_1198_jpg.rf.aa562bb4304e3a28543e0715ddf77190.txt
================================================
2 0.423828125 0.6552734375 0.42578125 0.6044921875 0.419921875 0.5908203125 0.4072265625 0.576171875 0.3583984375 0.5546875 0.3271484375 0.556640625 0.2822265625 0.58203125 0.2607421875 0.5859375 0.224609375 0.6181640625 0.23046875 0.6591796875 0.26171875 0.6943359375 0.3056640625 0.728515625 0.33984375 0.7353515625 0.4013671875 0.703125 0.412109375 0.6904296875 0.412109375 0.6669921875 0.423828125 0.6552734375


================================================
FILE: TumorDetection/valid/labels/meningioma_1202_jpg.rf.8006550acebd969ce2579747633630cc.txt
================================================
2 0.4326171875 0.14453125 0.3974609375 0.142578125 0.330078125 0.1904296875 0.328125 0.2177734375 0.3408203125 0.2421875 0.3837890625 0.263671875 0.41015625 0.2626953125 0.4453125 0.2451171875 0.462890625 0.1982421875 0.462890625 0.1787109375 0.4326171875 0.14453125


================================================
FILE: TumorDetection/valid/labels/meningioma_1205_jpg.rf.7a3551dd2ce9a35906dc1ae386c37e2d.txt
================================================
2 0.4365234375 0.1328125 0.3974609375 0.119140625 0.3125 0.1943359375 0.306640625 0.2119140625 0.330078125 0.2255859375 0.3642578125 0.2734375 0.408203125 0.2744140625 0.4365234375 0.259765625 0.455078125 0.2333984375 0.458984375 0.1611328125 0.4365234375 0.1328125


================================================
FILE: TumorDetection/valid/labels/meningioma_1208_jpg.rf.48b72738c07709590a0bde122c672a67.txt
================================================
2 0.3203125 0.5419921875 0.2998046875 0.515625 0.2646484375 0.501953125 0.2548828125 0.48828125 0.2216796875 0.48828125 0.216796875 0.5205078125 0.24609375 0.6025390625 0.2646484375 0.625 0.28515625 0.6279296875 0.306640625 0.6123046875 0.314453125 0.5673828125 0.326171875 0.5615234375 0.3203125 0.5419921875


================================================
FILE: TumorDetection/valid/labels/meningioma_1211_jpg.rf.7800f5eeb077d1398a44ce7edfbdec8c.txt
================================================
2 0.2703125 0.55625 0.1265625 0.13359375
2 0.3115234375 0.50390625 0.2294921875 0.501953125 0.208984375 0.5205078125 0.240234375 0.6083984375 0.265625 0.6142578125 0.3056640625 0.59375 0.32421875 0.5634765625 0.326171875 0.5244140625 0.3115234375 0.50390625


================================================
FILE: TumorDetection/valid/labels/meningioma_1215_jpg.rf.b507c5e7df19598a05db96b13cc71c75.txt
================================================
2 0.533203125 0.1943359375 0.53125 0.1572265625 0.5185546875 0.126953125 0.4462890625 0.115234375 0.3896484375 0.119140625 0.3232421875 0.1328125 0.2734375 0.1630859375 0.263671875 0.1943359375 0.291015625 0.2548828125 0.3525390625 0.302734375 0.39453125 0.3115234375 0.4287109375 0.306640625 0.4755859375 0.283203125 0.501953125 0.2607421875 0.533203125 0.1943359375


================================================
FILE: TumorDetection/valid/labels/meningioma_1223_jpg.rf.5ca4caf12a46da940adeba0c67a7f2b8.txt
================================================
2 0.6630859375 0.25390625 0.62890625 0.2744140625 0.6328125 0.3232421875 0.6494140625 0.333984375 0.666015625 0.3330078125 0.70703125 0.2998046875 0.708984375 0.2724609375 0.6943359375 0.2578125 0.6630859375 0.25390625


================================================
FILE: TumorDetection/valid/labels/meningioma_122_jpg.rf.b8f33c11af9abf5b2d3a732f436b74e9.txt
================================================
2 0.6572265625 0.771484375 0.6962890625 0.748046875 0.7255859375 0.744140625 0.755859375 0.7177734375 0.767578125 0.6787109375 0.755859375 0.6474609375 0.7138671875 0.60546875 0.6884765625 0.59765625 0.6435546875 0.59765625 0.638671875 0.5810546875 0.6123046875 0.55859375 0.5771484375 0.55859375 0.5615234375 0.56640625 0.537109375 0.6064453125 0.53515625 0.6220703125 0.533203125 0.6923828125 0.55078125 0.7216796875 0.5966796875 0.7578125 0.6572265625 0.771484375


================================================
FILE: TumorDetection/valid/labels/meningioma_1232_jpg.rf.461290cf338a9e9c13e2e11757d97c9a.txt
================================================
2 0.3212890625 0.458984375 0.35546875 0.4462890625 0.35546875 0.4326171875 0.37890625 0.4150390625 0.392578125 0.3818359375 0.380859375 0.3486328125 0.3447265625 0.3125 0.3173828125 0.30078125 0.2958984375 0.30078125 0.2705078125 0.318359375 0.255859375 0.3486328125 0.25390625 0.3818359375 0.267578125 0.4345703125 0.2958984375 0.44140625 0.3212890625 0.458984375


================================================
FILE: TumorDetection/valid/labels/meningioma_1241_jpg.rf.addad1eea8a59b7111e1a2c2ab052b97.txt
================================================
2 0.314453125 0.5283203125 0.3251953125 0.5234375 0.349609375 0.5322265625 0.3349609375 0.521484375 0.3056640625 0.521484375 0.2841796875 0.53125 0.263671875 0.5576171875 0.2734375 0.6259765625 0.2880859375 0.646484375 0.318359375 0.6552734375 0.33984375 0.6376953125 0.345703125 0.6181640625 0.349609375 0.5556640625 0.314453125 0.5283203125


================================================
FILE: TumorDetection/valid/labels/meningioma_1244_jpg.rf.d7d2600434c1faab0d4d353498543e2d.txt
================================================
2 0.4970703125 0.404296875 0.490234375 0.4365234375 0.4921875 0.4912109375 0.5478515625 0.517578125 0.568359375 0.5166015625 0.595703125 0.4775390625 0.58984375 0.4482421875 0.5341796875 0.4140625 0.4970703125 0.404296875


================================================
FILE: TumorDetection/valid/labels/meningioma_1245_jpg.rf.a4b95728201624d4cf56e70a12fa288f.txt
================================================
2 0.32421875 0.4580078125 0.30859375 0.4228515625 0.2861328125 0.400390625 0.2548828125 0.38671875 0.2421875 0.3994140625 0.220703125 0.4501953125 0.212890625 0.5107421875 0.224609375 0.5947265625 0.25 0.6416015625 0.259765625 0.6435546875 0.3125 0.6044921875 0.322265625 0.5693359375 0.326171875 0.4912109375 0.32421875 0.4580078125


================================================
FILE: TumorDetection/valid/labels/meningioma_1276_jpg.rf.7bae1a07294c81b208e6eb56c40ee365.txt
================================================
2 0.59765625 0.2880859375 0.59765625 0.2724609375 0.5888671875 0.26171875 0.5654296875 0.275390625 0.5498046875 0.275390625 0.5458984375 0.28515625 0.5341796875 0.26171875 0.5009765625 0.259765625 0.5 0.2470703125 0.4912109375 0.2421875 0.4755859375 0.240234375 0.4716796875 0.25390625 0.46875 0.2431640625 0.4609375 0.2431640625 0.46484375 0.3017578125 0.4697265625 0.29296875 0.4755859375 0.3203125 0.4951171875 0.3203125 0.5009765625 0.3125 0.5068359375 0.32421875 0.5439453125 0.328125 0.5419921875 0.31640625 0.5302734375 0.3203125 0.525390625 0.3115234375 0.5419921875 0.30859375 0.5478515625 0.29296875 0.55859375 0.3115234375 0.55078125 0.3212890625 0.560546875 0.3271484375 0.5625 0.2802734375 0.568359375 0.2822265625 0.5732421875 0.3125 0.59765625 0.2880859375


================================================
FILE: TumorDetection/valid/labels/meningioma_1285_jpg.rf.16719575be6242b178b7258a6f12e940.txt
================================================
2 0.48046875 0.4951171875 0.482421875 0.4619140625 0.474609375 0.4365234375 0.4482421875 0.41015625 0.404296875 0.3916015625 0.3984375 0.3525390625 0.3662109375 0.31640625 0.3095703125 0.298828125 0.2568359375 0.322265625 0.2265625 0.3759765625 0.23046875 0.4365234375 0.2802734375 0.48828125 0.3095703125 0.49609375 0.31640625 0.4794921875 0.3291015625 0.47265625 0.337890625 0.4853515625 0.33203125 0.4912109375 0.34765625 0.5029296875 0.34765625 0.5205078125 0.3564453125 0.529296875 0.3583984375 0.5078125 0.3662109375 0.517578125 0.376953125 0.5146484375 0.36328125 0.5224609375 0.37890625 0.5283203125 0.3779296875 0.541015625 0.41796875 0.5419921875 0.4326171875 0.5390625 0.4501953125 0.513671875 0.4697265625 0.5078125 0.48046875 0.4951171875


================================================
FILE: TumorDetection/valid/labels/meningioma_1287_jpg.rf.05edb14dc7377ce1acf214e91375d452.txt
================================================
2 0.3173828125 0.546875 0.3271484375 0.560546875 0.3486328125 0.564453125 0.3525390625 0.576171875 0.3916015625 0.572265625 0.4345703125 0.556640625 0.482421875 0.4755859375 0.478515625 0.4365234375 0.45703125 0.3818359375 0.408203125 0.3349609375 0.3935546875 0.30859375 0.3759765625 0.30078125 0.3642578125 0.283203125 0.3369140625 0.275390625 0.3193359375 0.27734375 0.25 0.3427734375 0.23828125 0.4033203125 0.216796875 0.4599609375 0.21875 0.4892578125 0.2353515625 0.5078125 0.2841796875 0.541015625 0.3173828125 0.546875


================================================
FILE: TumorDetection/valid/labels/meningioma_1299_jpg.rf.215ba961935cafb3806427e0c46c4b39.txt
================================================
2 0.5166015625 0.7734375 0.5302734375 0.7890625 0.5849609375 0.78515625 0.603515625 0.7724609375 0.634765625 0.7294921875 0.642578125 0.6806640625 0.626953125 0.6572265625 0.63671875 0.6357421875 0.6328125 0.6142578125 0.59765625 0.5712890625 0.5810546875 0.537109375 0.5439453125 0.5078125 0.5009765625 0.505859375 0.4599609375 0.52734375 0.4501953125 0.5234375 0.439453125 0.5478515625 0.439453125 0.5751953125 0.423828125 0.6064453125 0.419921875 0.6435546875 0.466796875 0.7431640625 0.4853515625 0.76171875 0.5166015625 0.7734375


================================================
FILE: TumorDetection/valid/labels/meningioma_1306_jpg.rf.9bb2c50d97af1fa023cb90661ffbad81.txt
================================================
2 0.3701171875 0.333984375 0.35546875 0.3525390625 0.357421875 0.3740234375 0.3662109375 0.3828125 0.384765625 0.3818359375 0.404296875 0.3583984375 0.400390625 0.3330078125 0.3916015625 0.326171875 0.3701171875 0.333984375


================================================
FILE: TumorDetection/valid/labels/meningioma_1315_jpg.rf.a6dae6a57801e04a0b060c32542d2276.txt
================================================
2 0.46484375 0.3955078125 0.451171875 0.3740234375 0.453125 0.3525390625 0.3798828125 0.302734375 0.3486328125 0.291015625 0.294921875 0.3251953125 0.298828125 0.3369140625 0.27734375 0.3623046875 0.27734375 0.3955078125 0.255859375 0.4052734375 0.236328125 0.4599609375 0.3251953125 0.515625 0.361328125 0.5166015625 0.4384765625 0.49609375 0.4609375 0.4755859375 0.46875 0.4541015625 0.46484375 0.3955078125


================================================
FILE: TumorDetection/valid/labels/meningioma_1317_jpg.rf.9afa257fe8f51e66ba43459099b5c27e.txt
================================================
2 0.5595703125 0.615234375 0.5390625 0.6298828125 0.533203125 0.6904296875 0.5498046875 0.701171875 0.58203125 0.7021484375 0.609375 0.6728515625 0.611328125 0.6337890625 0.5830078125 0.6171875 0.5595703125 0.615234375


================================================
FILE: TumorDetection/valid/labels/meningioma_1324_jpg.rf.4d5bff39778ccec03a572bb2f000e6b3.txt
================================================
2 0.5771484375 0.5703125 0.5517578125 0.568359375 0.4833984375 0.591796875 0.4609375 0.6240234375 0.484375 0.7060546875 0.4970703125 0.716796875 0.541015625 0.7216796875 0.58984375 0.6923828125 0.60546875 0.6533203125 0.60546875 0.5986328125 0.5771484375 0.5703125


================================================
FILE: TumorDetection/valid/labels/meningioma_132_jpg.rf.2133ab6c2698ba0e76b75643b73b3fa6.txt
================================================
2 0.3271484375 0.25390625 0.298828125 0.2685546875 0.2734375 0.3173828125 0.271484375 0.3388671875 0.28125 0.3603515625 0.3056640625 0.380859375 0.33203125 0.3818359375 0.359375 0.3681640625 0.384765625 0.2958984375 0.3642578125 0.26171875 0.3271484375 0.25390625


================================================
FILE: TumorDetection/valid/labels/meningioma_142_jpg.rf.bc16b000d35742306f2a1788dbb00a1d.txt
================================================
2 0.662109375 0.1552734375 0.6201171875 0.123046875 0.6044921875 0.123046875 0.591796875 0.1396484375 0.587890625 0.1611328125 0.6123046875 0.18359375 0.6376953125 0.1796875 0.662109375 0.1552734375


================================================
FILE: TumorDetection/valid/labels/meningioma_145_jpg.rf.af82b78e0d994f43a916bc2bf6527cf5.txt
================================================
2 0.6298828125 0.15234375 0.626953125 0.1787109375 0.634765625 0.1923828125 0.6552734375 0.208984375 0.671875 0.2099609375 0.689453125 0.1982421875 0.6875 0.1806640625 0.6455078125 0.150390625 0.6298828125 0.15234375


================================================
FILE: TumorDetection/valid/labels/meningioma_146_jpg.rf.29c24314852b50f3f04a4de7735c8261.txt
================================================
2 0.5322265625 0.15625 0.5048828125 0.15625 0.4833984375 0.166015625 0.47265625 0.1962890625 0.48046875 0.2314453125 0.5048828125 0.251953125 0.548828125 0.2548828125 0.591796875 0.2275390625 0.599609375 0.2080078125 0.595703125 0.1826171875 0.5322265625 0.15625


================================================
FILE: TumorDetection/valid/labels/meningioma_14_jpg.rf.1d5c9e2741a6af8a523f877d2e1d0050.txt
================================================
2 0.529296875 0.4091796875 0.5185546875 0.384765625 0.5029296875 0.384765625 0.4892578125 0.39453125 0.4541015625 0.38671875 0.4521484375 0.40234375 0.4404296875 0.400390625 0.4306640625 0.408203125 0.40625 0.3896484375 0.3876953125 0.35546875 0.375 0.3544921875 0.3671875 0.3974609375 0.369140625 0.4462890625 0.388671875 0.4951171875 0.3828125 0.5419921875 0.392578125 0.5439453125 0.4228515625 0.501953125 0.4853515625 0.46484375 0.52734375 0.4580078125 0.529296875 0.4091796875


================================================
FILE: TumorDetection/valid/labels/meningioma_160_jpg.rf.04837c098df7a5f0184c80af849b8401.txt
================================================
2 0.568359375 0.7470703125 0.5625 0.7333984375 0.5693359375 0.7265625 0.5751953125 0.734375 0.5791015625 0.716796875 0.6337890625 0.7265625 0.63671875 0.7060546875 0.6435546875 0.71484375 0.650390625 0.7080078125 0.650390625 0.6865234375 0.6318359375 0.66796875 0.6025390625 0.654296875 0.5341796875 0.64453125 0.5322265625 0.654296875 0.5126953125 0.6640625 0.470703125 0.7060546875 0.46484375 0.7705078125 0.484375 0.7919921875 0.5126953125 0.783203125 0.5322265625 0.76171875 0.5615234375 0.7578125 0.568359375 0.7470703125


================================================
FILE: TumorDetection/valid/labels/meningioma_179_jpg.rf.2f2fd849ee977f3d5c781dddb57725e8.txt
================================================
2 0.4873046859375 0.7403706390625 0.4941406265625 0.652934228125 0.4746093734375 0.5416515265624999 0.4423828140625 0.520076309375 0.4150390640625 0.522347384375 0.3769531265625 0.5779887359375 0.3613281265625 0.63249455 0.3769531265625 0.691542515625 0.4287109359375 0.744912790625 0.4873046859375 0.7403706390625


================================================
FILE: TumorDetection/valid/labels/meningioma_181_jpg.rf.a07e8ed8c0b3ee827c1e67d3fd60241b.txt
================================================
2 0.6304694109375 0.865234375 0.6755796578125 0.8671875 0.721763959375 0.8271484375 0.73250449375 0.8017578125 0.7647261000000001 0.7685546875 0.829169309375 0.6767578125 0.8152066140625 0.677734375 0.80124391875 0.6455078125 0.7615039390625 0.619140625 0.7249861203125 0.611328125 0.6712834453125 0.61328125 0.6455061609375 0.6171875 0.5982478046875 0.64453125 0.5692483609375001 0.6787109375 0.558507825 0.7080078125 0.5542116109374999 0.7841796875 0.567100253125 0.8271484375 0.6003959125 0.85546875 0.6304694109375 0.865234375


================================================
FILE: TumorDetection/valid/labels/meningioma_184_jpg.rf.a0cf04543cb6542980ef171f2a9a2fc7.txt
================================================
2 0.816636965625 0.6884765625 0.8116726062499999 0.6650390625 0.7756810078124999 0.642578125 0.728519603125 0.634765625 0.713626528125 0.64453125 0.6615007640624999 0.650390625 0.6428844203125 0.6630859375 0.6230269859375 0.6962890625 0.6155804484375 0.7373046875 0.62799134375 0.7763671875 0.6503309578125 0.7880859375 0.6565364046875 0.802734375 0.6912869140625 0.81640625 0.7458948578125 0.810546875 0.7496181265625 0.8212890625 0.7570646640625001 0.8212890625 0.7942973515625 0.7978515625 0.8067082484375 0.7490234375 0.8464231156249999 0.7001953125 0.816636965625 0.6884765625


================================================
FILE: TumorDetection/valid/labels/meningioma_199_jpg.rf.ab708827b573d2f27462da723fc4209e.txt
================================================
2 0.6090062265625 0.40625 0.66006064375 0.390625 0.7001748250000001 0.3544921875 0.7074683140625 0.2294921875 0.6454736671875 0.18359375 0.6041439015625001 0.1816406234375 0.562814140625 0.1542968765625 0.46556763593749995 0.1542968765625 0.4157288046875 0.2119140625 0.3938483390625 0.3623046875 0.410866475 0.4013671875 0.443687171875 0.4277343765625 0.6090062265625 0.40625


================================================
FILE: TumorDetection/valid/labels/meningioma_200_jpg.rf.1c8b789dd67a69f1e6fdd3c44f5fdb56.txt
================================================
2 0.5686333734375 0.6132812484375 0.5234866796875 0.5976562484375 0.4632910859375 0.5976562484375 0.4396428171875 0.6035156234375 0.41599454687500004 0.6367187515625 0.3751475390625 0.6308593765625 0.34397482031250004 0.6533203140625 0.3224763953125 0.7431640625 0.345049740625 0.7773437515625 0.394496121875 0.7910156234375 0.43749297499999995 0.7753906234375 0.4783399859375 0.796875 0.522411759375 0.8017578140625 0.561108925 0.7783203140625 0.5976562515625 0.7294921859375 0.59550640625 0.6474609375 0.5686333734375 0.6132812484375


================================================
FILE: TumorDetection/valid/labels/meningioma_209_jpg.rf.0fa4494baf364d68baff0ad0730987a9.txt
================================================
2 0.7148437484375 0.4074053078125 0.6640625015625 0.28028651718750003 0.6181640625 0.25006975468750003 0.5712890625 0.25215367031249997 0.5595703140625 0.2563215 0.5195312515625 0.29904175 0.5 0.3448878703125 0.4921874984375 0.394901821875 0.5039062515625 0.44699968749999996 0.546875 0.51785278125 0.5732421890625 0.5376499734375 0.6171874984375 0.54077584375 0.6533203140625 0.5397338890625 0.6953125015625 0.507433209375 0.716796875 0.4657549203125 0.7148437484375 0.4074053078125


================================================
FILE: TumorDetection/valid/labels/meningioma_211_jpg.rf.6131e84721c8c3c95f46a3c158e27b1d.txt
================================================
2 0.7750740828125 0.298828125 0.6863954765625 0.3564453125 0.6725754312500001 0.3857421875 0.6702720890625 0.4599609390625 0.6886988140625 0.5107421875 0.71518723125 0.5351562515625 0.7589507 0.5664062515625 0.822292565625 0.5771484390625 0.85453933125 0.5068359390625 0.8660560343749999 0.3232421875 0.8602976828125 0.306640625 0.7750740828125 0.298828125


================================================
FILE: TumorDetection/valid/labels/meningioma_213_jpg.rf.24e73bf7cbf7beef2c82aeba6d81623c.txt
================================================
2 0.5283203125 0.27563291093750003 0.5810546875 0.257120253125 0.599609375 0.231408228125 0.607421875 0.1984968359375 0.6513671875 0.1810126578125 0.671875 0.1347310125 0.6279296875 0.1151898734375 0.5146484375 0.1172468359375 0.4580078125 0.123417721875 0.384765625 0.1573575953125 0.384765625 0.17175632968750001 0.4267578125 0.230379746875 0.4912109375 0.27151898750000003 0.5283203125 0.27563291093750003


================================================
FILE: TumorDetection/valid/labels/meningioma_220_jpg.rf.6b95313f6e1d3412f2809acf3d1f87ce.txt
================================================
2 0.5224609359375 0.109375 0.4775390640625 0.12109375156249999 0.4482421875 0.11328124843750001 0.376953125 0.1533203125 0.373046875 0.1767578125 0.3857421875 0.193359375 0.4384765640625 0.2265625 0.486328125 0.2373046875 0.490234375 0.2216796875 0.529296875 0.1767578125 0.5351562484375 0.1416015640625 0.5224609359375 0.109375


================================================
FILE: TumorDetection/valid/labels/meningioma_228_jpg.rf.2728d4bdea8f0dd8d364538ad2f939b0.txt
================================================
2 0.6748046875 0.3873994515625 0.6435546875 0.361572821875 0.6103515625 0.361572821875 0.587890625 0.3945735140625 0.5703125 0.43761789999999995 0.572265625 0.5064889140625 0.5908203125 0.5480984828125 0.6181640625 0.5624466109374999 0.63671875 0.5610118 0.6630859375 0.5394896078125 0.681640625 0.5064889140625 0.685546875 0.4146608953125 0.6748046875 0.3873994515625


================================================
FILE: TumorDetection/valid/labels/meningioma_230_jpg.rf.5d11a5fb2c30e4f584732d516838c23c.txt
================================================
2 0.36437931875 0.427734375 0.4019657234375 0.4296875 0.41240639687499997 0.42578125 0.460433465625 0.38671875 0.4760944703125 0.3662109390625 0.4844470046875 0.2978515609375 0.4760944703125 0.2900390609375 0.46356566875 0.2607421890625 0.43955212968749996 0.2421875 0.40823012968750005 0.240234375 0.374819990625 0.25 0.3393217140625 0.27734375 0.3153081796875 0.3056640609375 0.3132200453125 0.3349609390625 0.336189515625 0.3720703109375 0.336189515625 0.4072265609375 0.36437931875 0.427734375


================================================
FILE: TumorDetection/valid/labels/meningioma_237_jpg.rf.446ceb32c2ae6efa17f77c23969dd9ee.txt
================================================
2 0.365234375 0.4541015625 0.3935546875 0.4921875 0.4189453125 0.5039062484375 0.4892578125 0.505859375 0.5107421875 0.494140625 0.5576171875 0.501953125 0.6142578125 0.478515625 0.697265625 0.3974609375 0.7148437515625 0.3681640625 0.7226562484375 0.3291015625 0.720703125 0.2802734375 0.705078125 0.2314453125 0.6962890625 0.2226562484375 0.6533203125 0.185546875 0.5888671875 0.15625 0.5283203125 0.150390625 0.5185546875 0.142578125 0.509765625 0.1494140625 0.4990234375 0.1796875 0.4443359375 0.1757812484375 0.3955078125 0.1914062484375 0.384765625 0.2021484375 0.3789062484375 0.2294921875 0.34375 0.2900390625 0.341796875 0.3271484375 0.353515625 0.3642578125 0.349609375 0.3896484375 0.365234375 0.4541015625


================================================
FILE: TumorDetection/valid/labels/meningioma_241_jpg.rf.1a532ae953b9a685d99d20cb4e59f433.txt
================================================
2 0.406960228125 0.3232421875 0.39630681874999996 0.3017578125 0.3206676125 0.2480468765625 0.2716619328125 0.2558593765625 0.235440340625 0.27734375 0.1875 0.3603515625 0.1853693203125 0.3857421875 0.1981534078125 0.4072265625 0.2567471578125 0.45703125 0.2769886359375 0.4560546875 0.32919034062499997 0.43359375 0.37393465937500003 0.390625 0.406960228125 0.3779296875 0.406960228125 0.3232421875


================================================
FILE: TumorDetection/valid/labels/meningioma_247_jpg.rf.4e654fe42faa65584d0d20217100cbe4.txt
================================================
2 0.8192202921874999 0.4169921875 0.7849763546874999 0.3583984359375 0.7362446015625 0.333984375 0.6940982203125 0.2929687515625 0.6177079046875 0.283203125 0.511024875 0.4013671875 0.48468338749999995 0.4541015640625 0.4925858359375 0.4970703125 0.5887322671875 0.5703125 0.620342053125 0.578125 0.6888299250000001 0.5703125 0.724390934375 0.5830078125 0.7652202406249999 0.5703125 0.8113178453125001 0.5361328125 0.8271227375000001 0.4794921875 0.8192202921874999 0.4169921875


================================================
FILE: TumorDetection/valid/labels/meningioma_252_jpg.rf.77f7e2a406897946dc65db7a52515195.txt
================================================
2 0.401015625 0.6201171859375 0.35953124999999997 0.5888671859375 0.3514648453125 0.5761718734375 0.33533202968750003 0.57421875 0.28693359531249996 0.5957031265625 0.2546679703125 0.5996093734375 0.16593750000000002 0.6396484359375 0.16593750000000002 0.6748046859375 0.182070309375 0.7060546859375 0.2270117203125 0.73828125 0.271953125 0.7451171859375 0.3053710953125 0.73828125 0.3606835953125 0.7109375 0.401015625 0.6748046859375 0.401015625 0.6201171859375


================================================
FILE: TumorDetection/valid/labels/meningioma_253_jpg.rf.e3e4cd09323e06fa5183418d371c3b1d.txt
================================================
2 0.5507812515625 0.75537109375 0.5507812515625 0.7115179109375 0.5371093734375 0.6764353625 0.4667968734375 0.6128482453124999 0.5136718734375 0.5514537875000001 0.4951171890625 0.43414651875 0.4863281265625 0.43743550937500003 0.4726562515625 0.525141878125 0.453125 0.5646097453125 0.4091796890625 0.600788621875 0.2841796890625 0.649027121875 0.2519531265625 0.685206 0.2421875 0.7180958875 0.25 0.7904536421875 0.2783203109375 0.8288251796875 0.3388671890625 0.8704857031250001 0.4101562515625 0.8803526703125 0.4423828109375 0.8748710187500001 0.5068359375 0.8332104953125 0.5351562515625 0.7970316171875 0.5507812515625 0.75537109375


================================================
FILE: TumorDetection/valid/labels/meningioma_257_jpg.rf.a6b03ddbfee225035425cf2886ab5ee3.txt
================================================
2 0.5255995421875 0.599609375 0.5524282890625 0.6484374984375 0.7060838406250001 0.640625 0.7658387765625 0.6123046890625 0.792667521875 0.5439453109375 0.7475464484375001 0.537109375 0.6853525328125001 0.4873046890625 0.6451094125 0.369140625 0.5695011296875 0.3730468765625 0.5475503328125 0.3515625015625 0.4597471640625 0.3417968765625 0.39145580625 0.349609375 0.363407571875 0.3779296890625 0.363407571875 0.5048828125 0.3975532484375 0.5654296890625 0.44511329843749997 0.599609375 0.5255995421875 0.599609375


================================================
FILE: TumorDetection/valid/labels/meningioma_274_jpg.rf.e9a45742a67a9992a8fee774e8feb986.txt
================================================
2 0.6926694624999999 0.5380859359375 0.8048624078125 0.5927734375 0.7853505921874999 0.4443359359375 0.7402295171875 0.4453125015625 0.6902304874999999 0.4912109359375 0.6926694624999999 0.5380859359375


================================================
FILE: TumorDetection/valid/labels/meningioma_277_jpg.rf.80a7f9edd64bd084710a1abf65be1fe5.txt
================================================
2 0.6328125 0.5693359375 0.6171875 0.5224609375 0.5810546875 0.482421875 0.5400390625 0.466796875 0.4990234375 0.470703125 0.455078125 0.4970703125 0.447265625 0.5458984375 0.451171875 0.5771484375 0.4765625 0.6533203125 0.5244140625 0.673828125 0.548828125 0.6748046875 0.5810546875 0.666015625 0.611328125 0.6357421875 0.626953125 0.6064453125 0.6328125 0.5693359375


================================================
FILE: TumorDetection/valid/labels/meningioma_284_jpg.rf.a29a234784d4b2a3011332b28523ccda.txt
================================================
2 0.5126953125 0.333984375 0.498046875 0.3623046875 0.49609375 0.3857421875 0.5068359375 0.396484375 0.52734375 0.3974609375 0.54296875 0.3896484375 0.55078125 0.3583984375 0.5400390625 0.341796875 0.5126953125 0.333984375


================================================
FILE: TumorDetection/valid/labels/meningioma_311_jpg.rf.59ad21f3f40ec08bf507e0c81b11bb27.txt
================================================
2 0.3349609375 0.48046875 0.3701171875 0.478515625 0.3984375 0.4521484375 0.419921875 0.3994140625 0.42578125 0.3720703125 0.421875 0.3251953125 0.3935546875 0.296875 0.3544921875 0.294921875 0.3232421875 0.3046875 0.291015625 0.3525390625 0.296875 0.3720703125 0.27734375 0.4091796875 0.28515625 0.4462890625 0.3349609375 0.48046875


================================================
FILE: TumorDetection/valid/labels/meningioma_326_jpg.rf.06ff6187aa7b2837ab2ca0f1f6409133.txt
================================================
2 0.5898437515625 0.3310546890625 0.5996093734375 0.3017578140625 0.5976562484375 0.2763671859375 0.5859375015625 0.2490234359375 0.5654296890625 0.2246093734375 0.5439453109375 0.216796875 0.5361328140625 0.2246093734375 0.5283203109375 0.220703125 0.5205078140625 0.2246093734375 0.5068359375 0.2128906265625 0.4736328140625 0.2128906265625 0.4433593734375 0.2373046890625 0.4257812484375 0.2880859375 0.4257812484375 0.3173828140625 0.4316406265625 0.3369140625 0.439453125 0.3505859375 0.4580078140625 0.3632812484375 0.4882812484375 0.3642578140625 0.5166015640625 0.359375 0.5576171859375 0.3632812484375 0.5693359375 0.357421875 0.5898437515625 0.3310546890625


================================================
FILE: TumorDetection/valid/labels/meningioma_327_jpg.rf.ae5ab1c26ed22d282ec65d161fcd34ba.txt
================================================
2 0.5576171875 0.49609375 0.5947265625 0.484375 0.62109375 0.4619140625 0.63671875 0.4345703125 0.63671875 0.3818359375 0.62109375 0.3564453125 0.5908203125 0.33203125 0.5458984375 0.326171875 0.5126953125 0.3359375 0.478515625 0.3701171875 0.48046875 0.4365234375 0.51171875 0.4794921875 0.5322265625 0.494140625 0.5576171875 0.49609375


================================================
FILE: TumorDetection/valid/labels/meningioma_332_jpg.rf.01427d915fbaf63631281218b34e67e0.txt
================================================
2 0.466796875 0.7861328125 0.47265625 0.7412109375 0.4306640625 0.693359375 0.3798828125 0.693359375 0.345703125 0.7255859375 0.341796875 0.7685546875 0.3525390625 0.78515625 0.390625 0.8056640625 0.38671875 0.8251953125 0.3955078125 0.837890625 0.4208984375 0.849609375 0.4375 0.8486328125 0.4609375 0.8310546875 0.46484375 0.8095703125 0.474609375 0.8037109375 0.466796875 0.7861328125


================================================
FILE: TumorDetection/valid/labels/meningioma_336_jpg.rf.03306e22d13f8033ad873bfcd32f019f.txt
================================================
2 0.4658203125 0.75 0.45703125 0.7880859375 0.4814453125 0.8046875 0.5 0.8076171875 0.53125 0.7919921875 0.525390625 0.7607421875 0.5029296875 0.74609375 0.4658203125 0.75


================================================
FILE: TumorDetection/valid/labels/meningioma_337_jpg.rf.38c432304d491168a629d5114392583f.txt
================================================
2 0.683832159375 0.4150390640625 0.6730120281249999 0.3857421859375 0.638387615625 0.3701171859375 0.631895540625 0.3427734359375 0.57887690625 0.3183593734375 0.566974765625 0.2939453140625 0.539924440625 0.28125 0.5009719765625 0.30078125 0.48906983593750003 0.3251953140625 0.4869058125 0.4365234359375 0.502053990625 0.5107421859375 0.5042180187499999 0.5849609359375 0.5442524921875 0.6230468734375 0.5615647 0.6230468734375 0.5713028171875 0.6142578140625 0.5723848312500001 0.5917968734375 0.5853689859375 0.6171875 0.6535357984375 0.6240234359375 0.677340084375 0.6064453140625 0.690324234375 0.5751953140625 0.6795041078125 0.4873046859375 0.683832159375 0.4150390640625


================================================
FILE: TumorDetection/valid/labels/meningioma_338_jpg.rf.d28d419f8acd2d514b7863bad0cbdaf7.txt
================================================
2 0.5068359375 0.583984375 0.564453125 0.5517578125 0.587890625 0.5087890625 0.595703125 0.4619140625 0.587890625 0.4287109375 0.568359375 0.4072265625 0.556640625 0.3720703125 0.5400390625 0.357421875 0.5107421875 0.34375 0.4697265625 0.34765625 0.408203125 0.4365234375 0.412109375 0.4775390625 0.435546875 0.5439453125 0.4677734375 0.576171875 0.5068359375 0.583984375


================================================
FILE: TumorDetection/valid/labels/meningioma_340_jpg.rf.85d39b3cfc1e67bc21ade37d06bed6fd.txt
================================================
2 0.572265625 0.4130859375 0.546875 0.3349609375 0.5361328125 0.318359375 0.5205078125 0.310546875 0.5068359375 0.310546875 0.4814453125 0.330078125 0.4482421875 0.322265625 0.3876953125 0.341796875 0.34765625 0.3740234375 0.328125 0.4130859375 0.33203125 0.4462890625 0.3623046875 0.47265625 0.376953125 0.4755859375 0.4013671875 0.521484375 0.4755859375 0.55859375 0.48828125 0.5576171875 0.5205078125 0.544921875 0.564453125 0.5048828125 0.576171875 0.4501953125 0.572265625 0.4130859375


================================================
FILE: TumorDetection/valid/labels/meningioma_350_jpg.rf.0e927d84952deb34014816419a8ae64f.txt
================================================
2 0.7326596484375 0.5771484359375 0.7162873656250001 0.5517578125 0.7121942953125 0.5263671875 0.674333390625 0.4765624984375 0.59042544375 0.466796875 0.502424421875 0.46875 0.4502377734375 0.5048828125 0.3970278515625 0.5869140625 0.40112092187500004 0.6494140625 0.42465607812500006 0.6933593734375 0.445121434375 0.701171875 0.477866 0.6972656265625 0.5003778875 0.7070312484375 0.55870414375 0.7099609375 0.6784264609374999 0.6914062484375 0.7040081515625001 0.6728515640625 0.7326596484375 0.6298828125 0.7326596484375 0.5771484359375


================================================
FILE: TumorDetection/valid/labels/meningioma_351_jpg.rf.b26dc300f6a32c38d64ebedd50e55c72.txt
================================================
2 0.5175513703125 0.7666015640625 0.5227686203124999 0.7734375 0.5332031234375 0.7578125 0.5321596765625001 0.7783203125 0.5478114296875 0.7890625 0.5916363421875 0.7949218765625 0.6646778671875 0.7792968765625 0.7241545390625 0.7177734359375 0.74293664375 0.6083984359375 0.7178938359375 0.5419921875 0.6646778671875 0.4804687515625 0.55615903125 0.46875 0.4852044109375 0.484375 0.41111943437500004 0.5400390640625 0.37772902343750003 0.6005859359375 0.392337328125 0.6923828125 0.42259738750000003 0.7167968765625 0.5227686203124999 0.7363281234375 0.5175513703125 0.7666015640625


================================================
FILE: TumorDetection/valid/labels/meningioma_356_jpg.rf.f31001a90c81278e3f9935663f2d9cdb.txt
================================================
2 0.4931640609375 0.3334618125 0.5224609390625 0.32527870000000003 0.59765625 0.254699359375 0.61328125 0.232195803125 0.625 0.179005575 0.5791015609375 0.1452502375 0.4990234390625 0.1247924578125 0.4658203109375 0.14115868125 0.421875 0.1912802421875 0.41796875 0.254699359375 0.431640625 0.30788958593750004 0.4755859390625 0.3334618125 0.4931640609375 0.3334618125


================================================
FILE: TumorDetection/valid/labels/meningioma_370_jpg.rf.dbec04b254ffcba58a97515c4fb72485.txt
================================================
2 0.4541015625 0.50390625 0.4208984375 0.4921875 0.3935546875 0.49609375 0.349609375 0.5400390625 0.34765625 0.5634765625 0.365234375 0.6103515625 0.390625 0.6142578125 0.416015625 0.5556640625 0.4482421875 0.533203125 0.46875 0.5302734375 0.4541015625 0.50390625


================================================
FILE: TumorDetection/valid/labels/meningioma_400_jpg.rf.2be92a05ad5b03f94b6ed8b649d80970.txt
================================================
2 0.74609375 0.3369140625 0.7177734375 0.3203125 0.6708984375 0.330078125 0.650390625 0.3564453125 0.6640625 0.4052734375 0.6884765625 0.427734375 0.708984375 0.4306640625 0.7294921875 0.423828125 0.74609375 0.4072265625 0.740234375 0.3916015625 0.74609375 0.3369140625


================================================
FILE: TumorDetection/valid/labels/meningioma_407_jpg.rf.563353fae3c19585fa4879834974c35f.txt
================================================
2 0.505859375 0.2763671875 0.5 0.2314453125 0.4462890625 0.216796875 0.4169921875 0.224609375 0.40625 0.2392578125 0.408203125 0.2548828125 0.4267578125 0.26953125 0.4443359375 0.271484375 0.4541015625 0.298828125 0.474609375 0.3037109375 0.4921875 0.2919921875 0.4873046875 0.283203125 0.4931640625 0.2890625 0.505859375 0.2763671875


================================================
FILE: TumorDetection/valid/labels/meningioma_408_jpg.rf.ac070051268f08ebd57d8502cdcc62d6.txt
================================================
2 0.5205078125 0.3125 0.51953125 0.3466796875 0.529296875 0.3466796875 0.529296875 0.2646484375 0.51953125 0.2353515625 0.529296875 0.2119140625 0.5517578125 0.20703125 0.5517578125 0.197265625 0.5224609375 0.193359375 0.5029296875 0.203125 0.4482421875 0.203125 0.4208984375 0.212890625 0.40234375 0.2333984375 0.4111328125 0.27734375 0.4384765625 0.2890625 0.4580078125 0.3125 0.5205078125 0.3125


================================================
FILE: TumorDetection/valid/labels/meningioma_432_jpg.rf.142ba8cd81a1ad3e876b9d2df40ac99b.txt
================================================
2 0.3935546875 0.6171875 0.3798828125 0.607421875 0.3544921875 0.60546875 0.3056640625 0.615234375 0.263671875 0.6357421875 0.2578125 0.6669921875 0.2724609375 0.689453125 0.3095703125 0.69921875 0.3232421875 0.71875 0.349609375 0.7177734375 0.3837890625 0.7109375 0.4140625 0.6865234375 0.4140625 0.6474609375 0.3935546875 0.6171875
2 0.3115234375 0.64453125 0.3076171875 0.66015625 0.2783203125 0.65234375 0.275390625 0.6650390625 0.287109375 0.6865234375 0.3115234375 0.6953125 0.3251953125 0.716796875 0.33984375 0.7177734375 0.4013671875 0.701171875 0.416015625 0.6767578125 0.412109375 0.6494140625 0.3115234375 0.64453125


================================================
FILE: TumorDetection/valid/labels/meningioma_438_jpg.rf.e8882ab433abd2ee8224b53ded6202cb.txt
================================================
2 0.580078125 0.3134765625 0.55859375 0.2646484375 0.5263671875 0.24609375 0.4853515625 0.197265625 0.41015625 0.2373046875 0.3984375 0.2568359375 0.400390625 0.2744140625 0.365234375 0.3173828125 0.36328125 0.3466796875 0.3896484375 0.38671875 0.4267578125 0.404296875 0.4423828125 0.421875 0.515625 0.4345703125 0.548828125 0.4208984375 0.546875 0.4013671875 0.57421875 0.3564453125 0.580078125 0.3134765625


================================================
FILE: TumorDetection/valid/labels/meningioma_43_jpg.rf.c5fbad305e63343f4b1773e5d501adff.txt
================================================
2 0.380859375 0.7509765625 0.4208984375 0.791015625 0.4384765625 0.78515625 0.4697265625 0.7890625 0.4833984375 0.771484375 0.5185546875 0.7578125 0.53125 0.7451171875 0.580078125 0.6435546875 0.556640625 0.5361328125 0.5009765625 0.505859375 0.4501953125 0.509765625 0.4140625 0.5419921875 0.396484375 0.5810546875 0.376953125 0.5947265625 0.361328125 0.6376953125 0.37109375 0.6572265625 0.357421875 0.6767578125 0.359375 0.7080078125 0.380859375 0.7509765625


================================================
FILE: TumorDetection/valid/labels/meningioma_450_jpg.rf.515eabc644cd7fbbc451a9430cb0c271.txt
================================================
2 0.4306640625 0.1953125 0.39453125 0.2177734375 0.400390625 0.2646484375 0.4208984375 0.279296875 0.4453125 0.2802734375 0.498046875 0.2451171875 0.4970703125 0.208984375 0.4755859375 0.212890625 0.4306640625 0.1953125


================================================
FILE: TumorDetection/valid/labels/meningioma_452_jpg.rf.c18054003dbe7d88c7abd13d59e7393a.txt
================================================
2 0.4638671875 0.49609375 0.4052734375 0.505859375 0.39453125 0.5205078125 0.396484375 0.5537109375 0.4150390625 0.568359375 0.4609375 0.5712890625 0.4892578125 0.541015625 0.5146484375 0.5390625 0.53125 0.5185546875 0.4638671875 0.49609375


================================================
FILE: TumorDetection/valid/labels/meningioma_454_jpg.rf.9d896f5e69af0c2576a586c91b62253f.txt
================================================
2 0.453125 0.5654296875 0.451171875 0.5830078125 0.4755859375 0.630859375 0.5068359375 0.638671875 0.541015625 0.6142578125 0.5546875 0.5849609375 0.55078125 0.5380859375 0.5419921875 0.53125 0.5166015625 0.529296875 0.4892578125 0.541015625 0.4697265625 0.515625 0.4462890625 0.50390625 0.4130859375 0.5078125 0.3984375 0.5244140625 0.4013671875 0.568359375 0.4287109375 0.5703125 0.462890625 0.5576171875 0.453125 0.5654296875


================================================
FILE: TumorDetection/valid/labels/meningioma_478_jpg.rf.842f69b991ae31cdee33ef6d08c55234.txt
================================================
2 0.7265625 0.2705078125 0.7265625 0.2431640625 0.69921875 0.1923828125 0.6689453125 0.166015625 0.6025390625 0.13671875 0.5322265625 0.125 0.501953125 0.1689453125 0.498046875 0.2607421875 0.5341796875 0.3046875 0.5703125 0.3173828125 0.5927734375 0.30859375 0.6923828125 0.306640625 0.71875 0.2861328125 0.7265625 0.2705078125


================================================
FILE: TumorDetection/valid/labels/meningioma_47_jpg.rf.0d318bff47238418444d43c002c2c905.txt
================================================
2 0.6015625 0.5361328125 0.5791015625 0.5078125 0.5556640625 0.501953125 0.5048828125 0.513671875 0.494140625 0.5263671875 0.486328125 0.5634765625 0.50390625 0.5908203125 0.501953125 0.6279296875 0.509765625 0.6318359375 0.5166015625 0.61328125 0.5576171875 0.62109375 0.59375 0.5791015625 0.6015625 0.5361328125


================================================
FILE: TumorDetection/valid/labels/meningioma_496_jpg.rf.987dbb12751d93d7f893d0d37257d99c.txt
================================================
2 0.5263671875 0.466796875 0.5673828125 0.46484375 0.5927734375 0.453125 0.630859375 0.4150390625 0.654296875 0.3740234375 0.65234375 0.3369140625 0.63671875 0.2978515625 0.5830078125 0.263671875 0.5439453125 0.26171875 0.4833984375 0.283203125 0.46484375 0.2978515625 0.44140625 0.3349609375 0.44140625 0.3818359375 0.462890625 0.4248046875 0.4912109375 0.453125 0.5263671875 0.466796875


================================================
FILE: TumorDetection/valid/labels/meningioma_499_jpg.rf.99ced13e332182c9382940d425e0e264.txt
================================================
2 0.357421875 0.3564453125 0.375 0.3173828125 0.373046875 0.2919921875 0.3603515625 0.275390625 0.3251953125 0.265625 0.283203125 0.2880859375 0.263671875 0.3291015625 0.2861328125 0.365234375 0.3271484375 0.373046875 0.357421875 0.3564453125


================================================
FILE: TumorDetection/valid/labels/meningioma_501_jpg.rf.89fc4fb7d64c4588946d015bbe4ce49c.txt
================================================
2 0.451171875 0.4658203125 0.3974609375 0.4296875 0.3720703125 0.427734375 0.3525390625 0.435546875 0.333984375 0.4736328125 0.3935546875 0.48046875 0.3974609375 0.48828125 0.40234375 0.4853515625 0.4052734375 0.5 0.4091796875 0.486328125 0.41796875 0.4990234375 0.408203125 0.5009765625 0.412109375 0.5439453125 0.4267578125 0.556640625 0.44140625 0.5556640625 0.462890625 0.5498046875 0.466796875 0.5263671875 0.4453125 0.4951171875 0.451171875 0.4658203125


================================================
FILE: TumorDetection/valid/labels/meningioma_503_jpg.rf.4b1927dc404e63ec1a63ae20ca889b5a.txt
================================================
2 0.708984375 0.3623046875 0.69921875 0.3369140625 0.6591796875 0.2890625 0.5966796875 0.294921875 0.5908203125 0.30859375 0.5869140625 0.291015625 0.5712890625 0.291015625 0.5625 0.3056640625 0.56640625 0.3369140625 0.546875 0.3564453125 0.544921875 0.3779296875 0.5712890625 0.40625 0.6220703125 0.431640625 0.6494140625 0.474609375 0.69921875 0.4814453125 0.7333984375 0.46875 0.75 0.4462890625 0.73046875 0.3857421875 0.708984375 0.3623046875


================================================
FILE: TumorDetection/valid/labels/meningioma_509_jpg.rf.8c179b9af1225f1f5ad521b9eb9ef3b3.txt
================================================
2 0.6318359375 0.41015625 0.6484375 0.3935546875 0.658203125 0.3740234375 0.66015625 0.3427734375 0.6318359375 0.3046875 0.5947265625 0.28125 0.5634765625 0.283203125 0.537109375 0.2998046875 0.541015625 0.3701171875 0.5517578125 0.38671875 0.6083984375 0.4140625 0.6318359375 0.41015625


================================================
FILE: TumorDetection/valid/labels/meningioma_516_jpg.rf.d0c5109527e9f3f694079e4e9355ac35.txt
================================================
2 0.3759765625 0.3145450359375 0.3515625015625 0.3509507125 0.349609375 0.41211224687500003 0.3935546890625 0.4776424640625 0.435546875 0.4936609625 0.470703125 0.455799059375 0.4765625015625 0.3655129828125 0.4501953109375 0.3087201296875 0.3759765625 0.3145450359375


================================================
FILE: TumorDetection/valid/labels/meningioma_536_jpg.rf.747123a0e95af24c44d486deaf727fdc.txt
================================================
2 0.5908203125 0.44921875 0.6015625 0.4404296875 0.603515625 0.4189453125 0.5830078125 0.39453125 0.5615234375 0.384765625 0.4658203125 0.388671875 0.4130859375 0.41015625 0.375 0.4423828125 0.361328125 0.4775390625 0.3701171875 0.48828125 0.37890625 0.4873046875 0.396484375 0.4560546875 0.4345703125 0.4296875 0.5556640625 0.431640625 0.5908203125 0.44921875


================================================
FILE: TumorDetection/valid/labels/meningioma_549_jpg.rf.2b62f4012f34e83fdfb0ac188771e854.txt
================================================
2 0.4248046875 0.189453125 0.3759765625 0.201171875 0.3359375 0.2353515625 0.337890625 0.2646484375 0.3642578125 0.296875 0.40625 0.3037109375 0.439453125 0.2822265625 0.458984375 0.2451171875 0.45703125 0.2099609375 0.4248046875 0.189453125


================================================
FILE: TumorDetection/valid/labels/meningioma_556_jpg.rf.7bccbc4cfc4d63e56ade8e3edcf93605.txt
================================================
2 0.6669921875 0.138671875 0.6357421875 0.126953125 0.6064453125 0.12890625 0.59375 0.1376953125 0.560546875 0.1904296875 0.56640625 0.2138671875 0.55859375 0.2431640625 0.5625 0.2861328125 0.5791015625 0.310546875 0.6201171875 0.333984375 0.677734375 0.3369140625 0.7490234375 0.302734375 0.779296875 0.2451171875 0.7412109375 0.193359375 0.6669921875 0.138671875


================================================
FILE: TumorDetection/valid/labels/meningioma_562_jpg.rf.c8a317a7417995899a34f894c9523495.txt
================================================
2 0.7001953125 0.443359375 0.7275390625 0.439453125 0.736328125 0.4287109375 0.74609375 0.3837890625 0.744140625 0.3447265625 0.73828125 0.3271484375 0.7060546875 0.302734375 0.6787109375 0.302734375 0.6513671875 0.314453125 0.62109375 0.3466796875 0.609375 0.3720703125 0.619140625 0.4130859375 0.6611328125 0.45703125 0.6767578125 0.458984375 0.7001953125 0.443359375


================================================
FILE: TumorDetection/valid/labels/meningioma_565_jpg.rf.436f153ae0c432014185002101529a09.txt
================================================
2 0.7197265625 0.478515625 0.73828125 0.4677734375 0.75390625 0.4443359375 0.763671875 0.3876953125 0.775390625 0.3623046875 0.7734375 0.3447265625 0.7529296875 0.31640625 0.7041015625 0.287109375 0.6572265625 0.28515625 0.6240234375 0.302734375 0.599609375 0.3388671875 0.599609375 0.3798828125 0.6259765625 0.4375 0.6845703125 0.474609375 0.7197265625 0.478515625


================================================
FILE: TumorDetection/valid/labels/meningioma_601_jpg.rf.c97e495c3e58d52ca6f9466d72f66518.txt
================================================
2 0.4208984375 0.46875 0.3916015625 0.455078125 0.3466796875 0.462890625 0.32421875 0.4873046875 0.322265625 0.5263671875 0.3583984375 0.5390625 0.388671875 0.5693359375 0.3916015625 0.58203125 0.419921875 0.5830078125 0.443359375 0.5693359375 0.455078125 0.5400390625 0.439453125 0.4873046875 0.4208984375 0.46875


================================================
FILE: TumorDetection/valid/labels/meningioma_614_jpg.rf.8b6217c6cc9fdaebaeb2771e1c71c472.txt
================================================
2 0.5185546875 0.4609375 0.4833984375 0.44921875 0.4658203125 0.458984375 0.43359375 0.4970703125 0.431640625 0.5107421875 0.4453125 0.5439453125 0.4599609375 0.55859375 0.5 0.5615234375 0.52734375 0.5478515625 0.541015625 0.5224609375 0.54296875 0.5009765625 0.5185546875 0.4609375


================================================
FILE: TumorDetection/valid/labels/meningioma_617_jpg.rf.64d2c7b7d10a36c41ef0772efa7f062c.txt
================================================
2 0.5390625 0.4736328125 0.5224609375 0.455078125 0.4697265625 0.44921875 0.4287109375 0.43359375 0.412109375 0.4580078125 0.404296875 0.5107421875 0.4169921875 0.521484375 0.4384765625 0.521484375 0.451171875 0.5439453125 0.45703125 0.5791015625 0.4716796875 0.58984375 0.48828125 0.5888671875 0.5078125 0.5810546875 0.5078125 0.5576171875 0.53515625 0.5244140625 0.5390625 0.4736328125


================================================
FILE: TumorDetection/valid/labels/meningioma_640_jpg.rf.c49efacb21a5ec7e1efec01e86923605.txt
================================================
2 0.5615234375 0.302734375 0.5966796875 0.2890625 0.619140625 0.2666015625 0.630859375 0.2392578125 0.625 0.1982421875 0.5986328125 0.177734375 0.5517578125 0.17578125 0.5068359375 0.189453125 0.4912109375 0.197265625 0.474609375 0.2216796875 0.478515625 0.2685546875 0.4951171875 0.2890625 0.5244140625 0.302734375 0.5615234375 0.302734375


================================================
FILE: TumorDetection/valid/labels/meningioma_652_jpg.rf.f1a511d2df8ad06444bd0743863fb44a.txt
================================================
2 0.701171875 0.5927734375 0.69921875 0.5810546875 0.6826171875 0.56640625 0.6376953125 0.560546875 0.6171875 0.5244140625 0.5888671875 0.509765625 0.5751953125 0.51171875 0.544921875 0.5419921875 0.55859375 0.5634765625 0.541015625 0.5849609375 0.546875 0.6123046875 0.53515625 0.6494140625 0.5478515625 0.662109375 0.572265625 0.6650390625 0.58984375 0.6259765625 0.6025390625 0.615234375 0.6630859375 0.587890625 0.701171875 0.5927734375


================================================
FILE: TumorDetection/valid/labels/meningioma_659_jpg.rf.b5f1d60fb87bb0bddcd7488bcb7cc9bd.txt
================================================
2 0.544921875 0.3623046875 0.515625 0.2490234375 0.4873046875 0.22265625 0.4208984375 0.2109375 0.3681640625 0.2265625 0.310546875 0.2568359375 0.30078125 0.2783203125 0.30859375 0.3115234375 0.30078125 0.3486328125 0.3125 0.3740234375 0.3818359375 0.43359375 0.4130859375 0.453125 0.439453125 0.4541015625 0.4794921875 0.4453125 0.5166015625 0.42578125 0.541015625 0.3974609375 0.544921875 0.3623046875


================================================
FILE: TumorDetection/valid/labels/meningioma_65_jpg.rf.190091d77c284bec2d2a19c2d594bbab.txt
================================================
2 0.5224609375 0.375 0.4853515625 0.37890625 0.4189453125 0.3515625 0.3984375 0.3955078125 0.396484375 0.4287109375 0.421875 0.4775390625 0.431640625 0.4794921875 0.4697265625 0.44921875 0.5576171875 0.443359375 0.53515625 0.3818359375 0.5224609375 0.375


================================================
FILE: TumorDetection/valid/labels/meningioma_667_jpg.rf.7b3840dc68d4c7f6edb9076b107db22a.txt
================================================
2 0.7158203125 0.64453125 0.7451171875 0.62109375 0.7509765625 0.625 0.748046875 0.6123046875 0.783203125 0.5576171875 0.6962890625 0.51171875 0.6630859375 0.515625 0.62109375 0.5498046875 0.619140625 0.5927734375 0.626953125 0.6044921875 0.6904296875 0.646484375 0.7158203125 0.64453125


================================================
FILE: TumorDetection/valid/labels/meningioma_688_jpg.rf.e946d6c6088c00542065b3b8090ff62b.txt
================================================
2 0.6728515625 0.4609375 0.734375 0.3818359375 0.734375 0.3291015625 0.7177734375 0.3046875 0.6845703125 0.28125 0.5654296875 0.259765625 0.548828125 0.2705078125 0.53125 0.3212890625 0.529296875 0.3583984375 0.544921875 0.4345703125 0.5712890625 0.4609375 0.6728515625 0.4609375


================================================
FILE: TumorDetection/valid/labels/meningioma_696_jpg.rf.e950f810702b1395b235c5cdc8f9384b.txt
================================================
2 0.4736328125 0.380859375 0.490234375 0.3544921875 0.484375 0.3291015625 0.48828125 0.2919921875 0.4228515625 0.259765625 0.3857421875 0.259765625 0.357421875 0.2978515625 0.349609375 0.3232421875 0.34765625 0.3896484375 0.3798828125 0.408203125 0.4423828125 0.400390625 0.4736328125 0.380859375


================================================
FILE: TumorDetection/valid/labels/meningioma_698_jpg.rf.fff3af97210a4990caca2d2fa374dc6a.txt
================================================
2 0.4171875 0.34375 0.184375 0.14375
2 0.48046875 0.2880859375 0.4697265625 0.28125 0.4033203125 0.28125 0.3603515625 0.31640625 0.3388671875 0.322265625 0.322265625 0.3486328125 0.3251953125 0.357421875 0.3447265625 0.349609375 0.3671875 0.3896484375 0.3818359375 0.400390625 0.41796875 0.4033203125 0.4658203125 0.392578125 0.49609375 0.3662109375 0.4921875 0.3193359375 0.48046875 0.2880859375


================================================
FILE: TumorDetection/valid/labels/meningioma_708_jpg.rf.afb26cf275398fa63ffe426a34326bd2.txt
================================================
2 0.56640625 0.3017578125 0.5546875 0.2587890625 0.4931640625 0.177734375 0.4365234375 0.173828125 0.3857421875 0.185546875 0.267578125 0.2529296875 0.2890625 0.3447265625 0.3076171875 0.3671875 0.3583984375 0.384765625 0.4169921875 0.3828125 0.470703125 0.3916015625 0.5087890625 0.3828125 0.5361328125 0.365234375 0.55859375 0.3310546875 0.56640625 0.3017578125


================================================
FILE: TumorDetection/valid/labels/meningioma_712_jpg.rf.181bd43be4347f25b904af5623d21bcc.txt
================================================
2 0.498046875 0.3115234375 0.4931640625 0.263671875 0.3662109375 0.30078125 0.3505859375 0.30859375 0.31640625 0.3486328125 0.3134765625 0.3671875 0.3330078125 0.375 0.3525390625 0.396484375 0.3828125 0.4052734375 0.4306640625 0.376953125 0.4423828125 0.357421875 0.4619140625 0.34765625 0.498046875 0.3115234375


================================================
FILE: TumorDetection/valid/labels/meningioma_728_jpg.rf.50b25c5f02105fade2a41b733bd29cd2.txt
================================================
2 0.7001953125 0.404296875 0.732421875 0.3642578125 0.720703125 0.3251953125 0.6630859375 0.27734375 0.6240234375 0.26953125 0.6123046875 0.27734375 0.6064453125 0.26953125 0.5595703125 0.2734375 0.5341796875 0.283203125 0.4970703125 0.279296875 0.48828125 0.2919921875 0.490234375 0.3896484375 0.5341796875 0.443359375 0.580078125 0.4501953125 0.5966796875 0.443359375 0.591796875 0.4365234375 0.6298828125 0.408203125 0.7001953125 0.404296875


================================================
FILE: TumorDetection/valid/labels/meningioma_735_jpg.rf.89b58d53957ab395f0a3916069c1b977.txt
================================================
2 0.5693359375 0.5390625 0.6025390625 0.525390625 0.6328125 0.4892578125 0.64453125 0.4501953125 0.62890625 0.3994140625 0.6083984375 0.380859375 0.5732421875 0.37109375 0.5439453125 0.373046875 0.5146484375 0.38671875 0.490234375 0.4111328125 0.4765625 0.4423828125 0.4765625 0.4814453125 0.48828125 0.5087890625 0.5400390625 0.541015625 0.5693359375 0.5390625


================================================
FILE: TumorDetection/valid/labels/meningioma_741_jpg.rf.7351a474636fe6a0f0b35845374a39bb.txt
================================================
2 0.4658203125 0.4765625 0.43359375 0.5107421875 0.435546875 0.5419921875 0.4736328125 0.5625 0.517578125 0.5615234375 0.55078125 0.5263671875 0.541015625 0.4990234375 0.5166015625 0.47265625 0.4658203125 0.4765625


================================================
FILE: TumorDetection/valid/labels/meningioma_742_jpg.rf.593293cfccc56c9bb359c85404e113bc.txt
================================================
2 0.5546875 0.4970703125 0.5205078125 0.466796875 0.4716796875 0.458984375 0.44140625 0.4775390625 0.4140625 0.5166015625 0.4140625 0.5322265625 0.4599609375 0.580078125 0.490234375 0.5830078125 0.5322265625 0.56640625 0.564453125 0.5380859375 0.56640625 0.5185546875 0.5546875 0.4970703125


================================================
FILE: TumorDetection/valid/labels/meningioma_745_jpg.rf.ff942852f698de3cedfd6569e42587a3.txt
================================================
2 0.587890625 0.5888671875 0.58203125 0.5322265625 0.5302734375 0.470703125 0.4658203125 0.4765625 0.4365234375 0.49609375 0.3984375 0.5419921875 0.4072265625 0.556640625 0.4208984375 0.5625 0.4365234375 0.5625 0.4501953125 0.541015625 0.46484375 0.5439453125 0.48828125 0.6376953125 0.501953125 0.6513671875 0.5263671875 0.6328125 0.5400390625 0.634765625 0.5693359375 0.619140625 0.587890625 0.5888671875


================================================
FILE: TumorDetection/valid/labels/meningioma_764_jpg.rf.f19fea213f59d95c947d6f8ede946a2f.txt
================================================
2 0.7119140625 0.587890625 0.6708984375 0.568359375 0.6318359375 0.568359375 0.5908203125 0.58203125 0.576171875 0.6005859375 0.57421875 0.6259765625 0.59765625 0.6806640625 0.6513671875 0.71875 0.677734375 0.7255859375 0.7197265625 0.69921875 0.748046875 0.6650390625 0.744140625 0.6181640625 0.7119140625 0.587890625


================================================
FILE: TumorDetection/valid/labels/meningioma_772_jpg.rf.032d80719301deb415065e7cc2b9306e.txt
================================================
2 0.6376953125 0.142578125 0.5966796875 0.119140625 0.5712890625 0.119140625 0.5439453125 0.13671875 0.537109375 0.1494140625 0.5390625 0.2197265625 0.546875 0.2431640625 0.5634765625 0.26171875 0.5927734375 0.275390625 0.619140625 0.2763671875 0.6416015625 0.271484375 0.6640625 0.2509765625 0.69140625 0.2001953125 0.6376953125 0.142578125


================================================
FILE: TumorDetection/valid/labels/meningioma_789_jpg.rf.1bb9fd320e1cdf0acaf638afdc6c0955.txt
================================================
2 0.724609375 0.5244140625 0.7080078125 0.49609375 0.6591796875 0.46875 0.5791015625 0.46875 0.5380859375 0.4921875 0.51171875 0.5302734375 0.501953125 0.5615234375 0.50390625 0.6142578125 0.529296875 0.6767578125 0.5595703125 0.71484375 0.6025390625 0.728515625 0.634765625 0.7275390625 0.69921875 0.6904296875 0.736328125 0.6298828125 0.751953125 0.5498046875 0.724609375 0.5244140625


================================================
FILE: TumorDetection/valid/labels/meningioma_79_jpg.rf.0d8ed387436af1ccbecc77149d1b098c.txt
================================================
2 0.5712890625 0.3515625 0.5361328125 0.345703125 0.5224609375 0.349609375 0.5 0.3740234375 0.498046875 0.4189453125 0.5048828125 0.439453125 0.55078125 0.4462890625 0.57421875 0.4345703125 0.595703125 0.4072265625 0.59765625 0.3916015625 0.5712890625 0.3515625


================================================
FILE: TumorDetection/valid/labels/meningioma_803_jpg.rf.ed6dc1faa416cc5e15683263e42dc714.txt
================================================
2 0.4013671875 0.52734375 0.4443359375 0.52734375 0.4619140625 0.541015625 0.4765625 0.5107421875 0.45703125 0.4912109375 0.4482421875 0.44921875 0.4013671875 0.44921875 0.3740234375 0.41796875 0.3388671875 0.41796875 0.31640625 0.4326171875 0.3154296875 0.451171875 0.2861328125 0.458984375 0.29296875 0.4814453125 0.2998046875 0.48046875 0.3095703125 0.49609375 0.4013671875 0.52734375


================================================
FILE: TumorDetection/valid/labels/meningioma_823_jpg.rf.ba2e94412b6bd773e0aa3dd7560f2331.txt
================================================
2 0.572265625 0.3466796875 0.5361328125 0.333984375 0.52734375 0.3095703125 0.4931640625 0.28515625 0.4755859375 0.3046875 0.431640625 0.3310546875 0.4267578125 0.357421875 0.4013671875 0.36328125 0.396484375 0.3740234375 0.4169921875 0.376953125 0.427734375 0.4111328125 0.4912109375 0.4609375 0.5 0.4599609375 0.5390625 0.4150390625 0.55078125 0.3662109375 0.5673828125 0.361328125 0.572265625 0.3466796875


================================================
FILE: TumorDetection/valid/labels/meningioma_835_jpg.rf.8cdc87734332846800ce1774edfdfd5b.txt
================================================
2 0.5986328125 0.51953125 0.56640625 0.5458984375 0.560546875 0.5947265625 0.5791015625 0.62109375 0.638671875 0.6357421875 0.673828125 0.6044921875 0.666015625 0.5576171875 0.6376953125 0.529296875 0.5986328125 0.51953125


================================================
FILE: TumorDetection/valid/labels/meningioma_836_jpg.rf.ed2528da9d3789aa436154d02c60f7bf.txt
================================================
2 0.6181640625 0.52734375 0.578125 0.5634765625 0.572265625 0.5986328125 0.5888671875 0.6171875 0.62890625 0.6181640625 0.642578125 0.6123046875 0.66015625 0.5615234375 0.6416015625 0.537109375 0.6181640625 0.52734375


================================================
FILE: TumorDetection/valid/labels/meningioma_840_jpg.rf.c0d963f29e0bee565588a01d34e0fca8.txt
================================================
2 0.4248046875 0.453125 0.4052734375 0.44921875 0.373046875 0.4755859375 0.35546875 0.5068359375 0.353515625 0.5283203125 0.3935546875 0.578125 0.4140625 0.5791015625 0.4423828125 0.57421875 0.455078125 0.5634765625 0.462890625 0.4970703125 0.4248046875 0.453125


================================================
FILE: TumorDetection/valid/labels/meningioma_848_jpg.rf.eb602cae880b865a51e899ba88593e67.txt
================================================
2 0.5625 0.4189453125 0.5244140625 0.3984375 0.4794921875 0.3984375 0.462890625 0.4189453125 0.46484375 0.4365234375 0.453125 0.4541015625 0.45703125 0.4892578125 0.53515625 0.4990234375 0.556640625 0.4794921875 0.556640625 0.4521484375 0.56640625 0.4326171875 0.5625 0.4189453125


================================================
FILE: TumorDetection/valid/labels/meningioma_849_jpg.rf.03cac650e89f33cf6a0e1d4f57c745ce.txt
================================================
2 0.5693359375 0.69921875 0.55078125 0.7158203125 0.544921875 0.7431640625 0.5595703125 0.76171875 0.583984375 0.7646484375 0.599609375 0.7509765625 0.603515625 0.7294921875 0.5810546875 0.69921875 0.5693359375 0.69921875


================================================
FILE: TumorDetection/valid/labels/meningioma_860_jpg.rf.090b2389cdfec5fada884008555b1a82.txt
================================================
2 0.6982421875 0.404296875 0.7421875 0.3583984375 0.73046875 0.3115234375 0.70703125 0.2783203125 0.70703125 0.2412109375 0.6962890625 0.232421875 0.6591796875 0.220703125 0.6259765625 0.224609375 0.599609375 0.2431640625 0.587890625 0.2724609375 0.587890625 0.3115234375 0.59765625 0.3369140625 0.6435546875 0.404296875 0.6982421875 0.404296875


================================================
FILE: TumorDetection/valid/labels/meningioma_865_jpg.rf.870d1d3447e552366f47eccf0c57a8f7.txt
================================================
2 0.37890625 0.4130859375 0.375 0.3818359375 0.3828125 0.3271484375 0.3642578125 0.302734375 0.3271484375 0.28515625 0.3115234375 0.28515625 0.2890625 0.2998046875 0.26953125 0.3369140625 0.2421875 0.3623046875 0.23046875 0.3974609375 0.23046875 0.4658203125 0.2724609375 0.48828125 0.3046875 0.4931640625 0.3291015625 0.48828125 0.345703125 0.4716796875 0.373046875 0.4345703125 0.37890625 0.4130859375


================================================
FILE: TumorDetection/valid/labels/meningioma_871_jpg.rf.3783e22c8481fa2bfd8e0ce0a8b1b24a.txt
================================================
2 0.5595703125 0.509765625 0.603515625 0.4873046875 0.623046875 0.4228515625 0.6083984375 0.396484375 0.5732421875 0.37890625 0.5537109375 0.37890625 0.5244140625 0.390625 0.505859375 0.4150390625 0.501953125 0.4716796875 0.5146484375 0.49609375 0.5361328125 0.5078125 0.5595703125 0.509765625


================================================
FILE: TumorDetection/valid/labels/meningioma_875_jpg.rf.4720e8a8acbf9c3398270b4c74aa7f0d.txt
================================================
2 0.521484375 0.1708984375 0.4951171875 0.15625 0.4677734375 0.15625 0.4111328125 0.173828125 0.3984375 0.1904296875 0.40625 0.2255859375 0.4345703125 0.251953125 0.474609375 0.2568359375 0.5009765625 0.25 0.521484375 0.2314453125 0.529296875 0.2080078125 0.521484375 0.1708984375


================================================
FILE: TumorDetection/valid/labels/meningioma_878_jpg.rf.4487f4ab0d6afba1cd025d0052a1ef67.txt
================================================
2 0.6748046875 0.330078125 0.6259765625 0.341796875 0.5693359375 0.330078125 0.556640625 0.3408203125 0.55078125 0.3564453125 0.56640625 0.4345703125 0.5810546875 0.447265625 0.611328125 0.4521484375 0.6298828125 0.44921875 0.689453125 0.3837890625 0.69921875 0.3544921875 0.6748046875 0.330078125


================================================
FILE: TumorDetection/valid/labels/meningioma_884_jpg.rf.42401315a4eb27916b3b79b99e835899.txt
================================================
2 0.6337890625 0.3359375 0.5517578125 0.3359375 0.52734375 0.3681640625 0.529296875 0.4130859375 0.5390625 0.4365234375 0.5595703125 0.451171875 0.583984375 0.4521484375 0.6162109375 0.439453125 0.654296875 0.3994140625 0.654296875 0.3623046875 0.6337890625 0.3359375


================================================
FILE: TumorDetection/valid/labels/meningioma_887_jpg.rf.f4aea839efb9519489a36649a8f89b05.txt
================================================
2 0.2978515625 0.513671875 0.3310546875 0.51171875 0.341796875 0.5419921875 0.3603515625 0.5546875 0.4150390625 0.53125 0.419921875 0.5185546875 0.408203125 0.4931640625 0.4716796875 0.45703125 0.48046875 0.4345703125 0.46875 0.4208984375 0.470703125 0.3876953125 0.453125 0.3369140625 0.4228515625 0.3125 0.3447265625 0.3046875 0.2958984375 0.310546875 0.275390625 0.3505859375 0.275390625 0.3916015625 0.255859375 0.4423828125 0.271484375 0.4853515625 0.2978515625 0.513671875


================================================
FILE: TumorDetection/valid/labels/meningioma_894_jpg.rf.f5e917b9f2acd2761973efb74826bcea.txt
================================================
2 0.3212890625 0.251953125 0.2451171875 0.25 0.1826171875 0.26953125 0.15625 0.3076171875 0.138671875 0.3505859375 0.134765625 0.3857421875 0.150390625 0.4248046875 0.1884765625 0.453125 0.2138671875 0.451171875 0.2236328125 0.431640625 0.234375 0.4599609375 0.2451171875 0.470703125 0.259765625 0.4716796875 0.302734375 0.4423828125 0.34375 0.3330078125 0.349609375 0.2783203125 0.3212890625 0.251953125


================================================
FILE: TumorDetection/valid/labels/meningioma_8_jpg.rf.cfef3b130d130b6cb51385d8589cc45f.txt
================================================
2 0.587890625 0.5537109375 0.5859375 0.5654296875 0.6123046875 0.595703125 0.6982421875 0.58984375 0.71484375 0.5791015625 0.732421875 0.5517578125 0.7265625 0.5009765625 0.73046875 0.4560546875 0.73828125 0.4462890625 0.73046875 0.4306640625 0.7138671875 0.4140625 0.6953125 0.4169921875 0.7021484375 0.427734375 0.6845703125 0.42578125 0.6611328125 0.41015625 0.6494140625 0.390625 0.5927734375 0.404296875 0.533203125 0.4794921875 0.53125 0.5068359375 0.517578125 0.5224609375 0.5791015625 0.541015625 0.587890625 0.5537109375


================================================
FILE: TumorDetection/valid/labels/meningioma_906_jpg.rf.ba523c9e3efbfe87629e4851889cd4a8.txt
================================================
2 0.5986328125 0.57421875 0.5654296875 0.56640625 0.5400390625 0.578125 0.533203125 0.5888671875 0.5390625 0.6455078125 0.52734375 0.6982421875 0.5400390625 0.7265625 0.5654296875 0.740234375 0.5859375 0.7392578125 0.626953125 0.6982421875 0.63671875 0.6513671875 0.61328125 0.5869140625 0.5986328125 0.57421875


================================================
FILE: TumorDetection/valid/labels/meningioma_908_jpg.rf.11ee7d1420dfa545edd3223c85debd32.txt
================================================
2 0.4248046875 0.4375 0.3623046875 0.46875 0.33984375 0.4892578125 0.34375 0.5341796875 0.3681640625 0.5546875 0.396484375 0.5576171875 0.42578125 0.5419921875 0.451171875 0.4833984375 0.443359375 0.4560546875 0.4248046875 0.4375


================================================
FILE: TumorDetection/valid/labels/meningioma_911_jpg.rf.58a107efd4513c76e7e82c00e6cdf40f.txt
================================================
2 0.646484375 0.4794921875 0.5732421875 0.455078125 0.5322265625 0.4609375 0.498046875 0.4990234375 0.49609375 0.5185546875 0.5576171875 0.5234375 0.5625 0.5517578125 0.583984375 0.5908203125 0.61328125 0.6103515625 0.6328125 0.5986328125 0.630859375 0.5654296875 0.658203125 0.5205078125 0.646484375 0.4794921875


================================================
FILE: TumorDetection/valid/labels/meningioma_916_jpg.rf.486f441b6c2dfcf3cb01db035ac92889.txt
================================================
2 0.7080078125 0.2421875 0.6767578125 0.22265625 0.611328125 0.2646484375 0.59765625 0.3154296875 0.6484375 0.4423828125 0.6767578125 0.453125 0.705078125 0.4521484375 0.75390625 0.4365234375 0.765625 0.3505859375 0.751953125 0.2900390625 0.7080078125 0.2421875


================================================
FILE: TumorDetection/valid/labels/meningioma_921_jpg.rf.8c06075a71d54ee1e4f321aef96662ab.txt
================================================
2 0.498046875 0.3134765625 0.501953125 0.2900390625 0.4912109375 0.2578125 0.4404296875 0.2578125 0.4140625 0.2705078125 0.412109375 0.2919921875 0.4365234375 0.330078125 0.4697265625 0.333984375 0.498046875 0.3134765625


================================================
FILE: TumorDetection/valid/labels/meningioma_92_jpg.rf.419f05266e9cec2a0cee373d87679329.txt
================================================
2 0.6142578125 0.607421875 0.5888671875 0.59765625 0.5498046875 0.599609375 0.541015625 0.6123046875 0.53515625 0.6611328125 0.5478515625 0.685546875 0.587890625 0.6904296875 0.615234375 0.6748046875 0.6123046875 0.650390625 0.6640625 0.6494140625 0.6142578125 0.607421875


================================================
FILE: TumorDetection/valid/labels/meningioma_945_jpg.rf.321c5e911f77a86199621386abaf461d.txt
================================================
2 0.62109375 0.2763671875 0.630859375 0.2490234375 0.630859375 0.2001953125 0.5966796875 0.146484375 0.5341796875 0.142578125 0.5107421875 0.15234375 0.4921875 0.2001953125 0.4931640625 0.27734375 0.5419921875 0.298828125 0.5927734375 0.3046875 0.62109375 0.2763671875


================================================
FILE: TumorDetection/valid/labels/meningioma_950_jpg.rf.76630619b029f672853d569224775be8.txt
================================================
2 0.5693359375 0.2734375 0.54296875 0.2978515625 0.541015625 0.3173828125 0.5634765625 0.353515625 0.59765625 0.3583984375 0.6171875 0.3408203125 0.619140625 0.2998046875 0.6005859375 0.271484375 0.5693359375 0.2734375


================================================
FILE: TumorDetection/valid/labels/meningioma_960_jpg.rf.c012468e3499961dd54bd347dca969ab.txt
================================================
2 0.7421875 0.6650390625 0.736328125 0.6357421875 0.6943359375 0.615234375 0.6435546875 0.60546875 0.6181640625 0.609375 0.587890625 0.6435546875 0.583984375 0.6826171875 0.58984375 0.6943359375 0.6201171875 0.712890625 0.67578125 0.7177734375 0.6884765625 0.69921875 0.7197265625 0.693359375 0.734375 0.6826171875 0.7421875 0.6650390625


================================================
FILE: TumorDetection/valid/labels/meningioma_964_jpg.rf.dc11bf8d914d64dafa7915955dab2a0f.txt
================================================
2 0.47265625 0.4306640625 0.4892578125 0.443359375 0.5205078125 0.443359375 0.5615234375 0.431640625 0.62109375 0.3759765625 0.63671875 0.3466796875 0.63671875 0.3056640625 0.611328125 0.2861328125 0.59375 0.2431640625 0.5107421875 0.201171875 0.4853515625 0.23828125 0.443359375 0.2587890625 0.4296875 0.2841796875 0.42578125 0.3330078125 0.435546875 0.3662109375 0.462890625 0.3955078125 0.47265625 0.4306640625


================================================
FILE: TumorDetection/valid/labels/meningioma_965_jpg.rf.dce38666413e6b6147fb853df960c294.txt
================================================
2 0.4755859375 0.447265625 0.5185546875 0.45703125 0.5615234375 0.44140625 0.615234375 0.4052734375 0.6328125 0.3662109375 0.634765625 0.3291015625 0.626953125 0.3037109375 0.611328125 0.2880859375 0.603515625 0.2568359375 0.5810546875 0.228515625 0.5498046875 0.21875 0.5087890625 0.22265625 0.4873046875 0.248046875 0.4521484375 0.2578125 0.439453125 0.2724609375 0.4375 0.3447265625 0.4453125 0.3681640625 0.46484375 0.3916015625 0.4755859375 0.447265625


================================================
FILE: TumorDetection/valid/labels/meningioma_984_jpg.rf.e983f95359483021910289163782e42a.txt
================================================
2 0.5986328125 0.28515625 0.62890625 0.2314453125 0.626953125 0.2080078125 0.603515625 0.1591796875 0.5732421875 0.14453125 0.5458984375 0.1484375 0.5068359375 0.169921875 0.48046875 0.2119140625 0.4765625 0.2373046875 0.48828125 0.2705078125 0.5380859375 0.296875 0.5986328125 0.28515625


================================================
FILE: TumorDetection/valid/labels/meningioma_991_jpg.rf.931c156f35137b76b908565975f019af.txt
================================================
2 0.5517578125 0.595703125 0.55078125 0.6220703125 0.5615234375 0.640625 0.6005859375 0.64453125 0.62890625 0.6240234375 0.638671875 0.5673828125 0.654296875 0.5498046875 0.6455078125 0.52734375 0.5810546875 0.5 0.5205078125 0.501953125 0.498046875 0.5263671875 0.4921875 0.5595703125 0.50390625 0.5830078125 0.5224609375 0.59375 0.5517578125 0.595703125


================================================
FILE: TumorDetection/valid/labels/meningioma_994_jpg.rf.9001dc4787284f16f6b1ae12e5e8ffd7.txt
================================================
2 0.642578125 0.4912109375 0.5927734375 0.46484375 0.5341796875 0.4765625 0.5087890625 0.509765625 0.494140625 0.5146484375 0.513671875 0.5615234375 0.546875 0.5830078125 0.5595703125 0.609375 0.57421875 0.6123046875 0.609375 0.6025390625 0.626953125 0.5537109375 0.646484375 0.5302734375 0.642578125 0.4912109375


================================================
FILE: TumorDetection/valid/labels/meningioma_996_jpg.rf.60e56ad8f2e53ec66cb63d5f0bca3b50.txt
================================================
2 0.5986328125 0.306640625 0.6728515625 0.291015625 0.7109375 0.2509765625 0.705078125 0.1962890625 0.6572265625 0.158203125 0.5693359375 0.12890625 0.53515625 0.1552734375 0.52734375 0.1748046875 0.5234375 0.2158203125 0.53125 0.2529296875 0.5478515625 0.27734375 0.5986328125 0.306640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1002_jpg.rf.eb7141382aa74daadf933293752fc0e9.txt
================================================
0 0.8183593734375 0.4382152296875 0.76953125 0.27374926718750003 0.6884765640625 0.15364583125 0.5654296859375 0.0454445453125 0.4365234359375 0.0216402578125 0.3193359359375 0.0779049296875 0.16796875 0.2650931640625 0.12109375 0.4122469203125 0.1171875 0.570220803125 0.140625 0.6849141734375 0.2158203140625 0.8418060468749999 0.3544921859375 0.9738116203124999 0.52734375 0.9878777890625001 0.6787109359375 0.9067268171875 0.75390625 0.8017715687500001 0.8144531265625 0.635141578125 0.8183593734375 0.4382152296875


================================================
FILE: TumorDetection/valid/labels/no_tumor_1007_jpg.rf.353ec7f8439cfce18e6427ff90b0227d.txt
================================================
0 0.8476562515625 0.391840371875 0.7929687484375 0.21947069375 0.6943359359375 0.118253615625 0.6044921875 0.07616334531249999 0.4501953109375 0.058124659375 0.3251953109375 0.1222622140625 0.2167968765625 0.2675738609375 0.1601562515625 0.45998652343749996 0.1367187484375 0.6564077859374999 0.1640625015625 0.7766656984375 0.2666015609375 0.857839790625 0.3876953109375 0.91195585625 0.5195312515625 0.933000990625 0.7099609359375 0.863852690625 0.8164062515625 0.7626356109375 0.8476562515625 0.6664292796875 0.8476562515625 0.391840371875


================================================
FILE: TumorDetection/valid/labels/no_tumor_1031_jpg.rf.1813fcf1fb0b41bdb976e8d18fe1883f.txt
================================================
0 0.8746603281249999 0.5634765640625 0.8704144015624999 0.4501953109375 0.7939877734375 0.2158203109375 0.7037618890624999 0.09960937343750001 0.6549337640624999 0.07031249843750001 0.5678923234375 0.0527343734375 0.5371093765625 0.06347656406249999 0.5105723531249999 0.107421875 0.47872792343750004 0.0566406265625 0.42353090781249997 0.0566406265625 0.3141983671875 0.10839843593750001 0.214419159375 0.2607421859375 0.1677139953125 0.4306640625 0.16134511093749998 0.6455078140625 0.256878396875 0.8330078140625 0.31950577343750003 0.890625 0.40654721250000003 0.9316406265625 0.4978345796875 0.921875 0.5625849171875 0.9365234359375 0.6273352562500001 0.921875 0.68677819375 0.8828124984375 0.8215862749999999 0.7255859375 0.8746603281249999 0.5634765640625
4 0.4840353234375 0.2880859375 0.46068274374999996 0.2607421859375 0.4383916421875 0.2421875015625 0.42565387499999996 0.234375 0.41079314062499994 0.234375 0.390625 0.2470703109375 0.38637907656250003 0.2685546890625 0.3980553671875 0.2851562484375 0.42353090781249997 0.294921875 0.44582200937499994 0.3193359375 0.4500679328125 0.3291015640625 0.44582200937499994 0.3720703109375 0.4585597828125 0.3994140625 0.4670516328125 0.4033203109375 0.48191236249999997 0.3876953109375 0.4882812515625 0.3564453109375 0.4904042125 0.3154296890625 0.4840353234375 0.2880859375
4 0.5944293484375 0.3095703109375 0.645380434375 0.2705078140625 0.647503396875 0.2509765640625 0.63901155 0.2294921859375 0.6315811828125 0.220703125 0.6145974875 0.2148437515625 0.5976137921875 0.2148437515625 0.5615234390625 0.2402343734375 0.5010190234375 0.3076171859375 0.4988960578125 0.3310546890625 0.5095108703125 0.3662109375 0.550908628125 0.404296875 0.56046195625 0.4033203109375 0.57107676875 0.3974609375 0.585937496875 0.3740234359375 0.581691575 0.3349609375 0.5944293484375 0.3095703109375
4 0.42246942968749995 0.7822265640625 0.4182235078125 0.7548828140625 0.39487092343749997 0.7099609375 0.390625 0.6865234359375 0.39274796093750003 0.6650390625 0.40760869531250005 0.6337890625 0.3980553671875 0.6152343734375 0.36833389843750003 0.6035156265625 0.34498131875000004 0.580078125 0.3258746625 0.578125 0.31632133281249997 0.5986328140625 0.31207540625 0.6396484359375 0.31632133281249997 0.6748046890625 0.324813178125 0.6982421859375 0.341796875 0.7080078140625 0.341796875 0.7412109375 0.3587805703125 0.7587890625 0.3587805703125 0.7666015640625 0.3736413046875 0.7763671859375 0.36939538125000004 0.7822265640625 0.38213315 0.7919921859375 0.3778872265625 0.8134765640625 0.38531759375 0.8203124984375 0.405485734375 0.8212890625 0.416100540625 0.8115234359375 0.42034646875000004 0.8017578140625 0.416100540625 0.7919921859375 0.42246942968749995 0.7822265640625
4 0.6920855984375 0.7021484359375 0.7196841015625 0.6533203109375 0.717561140625 0.5888671859375 0.7133152171875 0.5751953109375 0.6931470765625 0.560546875 0.6549337640624999 0.578125 0.617781928125 0.6103515640625 0.61141304375 0.6259765640625 0.615658965625 0.7041015640625 0.5986752703125 0.7548828140625 0.581691575 0.7783203109375 0.575322690625 0.8095703109375 0.5774456515625 0.8291015640625 0.585937496875 0.8447265640625 0.5923063859375 0.8447265640625 0.6347656234375 0.8076171859375 0.645380434375 0.7607421859375 0.6751019031250001 0.7158203109375 0.6920855984375 0.7021484359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1065_jpg.rf.1ec3b70d96fb8b78a04d1d441e06b212.txt
================================================
0 0.951300934375 0.6376953125 0.9558094718750001 0.5185546875 0.9062155859375001 0.3095703125 0.8047735421875 0.1552734390625 0.6345763343749999 0.06640625156249999 0.44296358593749996 0.0585937484375 0.28516485468749997 0.11132812656249999 0.2164096921875 0.1767578125 0.1352560578125 0.2919921875 0.0766451015625 0.4658203125 0.0698822953125 0.6201171875 0.1059505765625 0.7724609390625 0.20851975312499998 0.8671875 0.36631848906250003 0.9335937484375 0.5545498328125 0.9462890609375 0.7270013078125 0.9140625 0.799137871875 0.8769531265625 0.890435709375 0.7998046875 0.951300934375 0.6376953125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1067_jpg.rf.6759483b515ce59d018e993929253c0c.txt
================================================
0 0.9051094890625 0.8251953125 0.9428223843750001 0.7646484375 0.9805352796875001 0.6669921875 0.992320559375 0.4814453125 0.9781782234374999 0.4033203125 0.945179440625 0.3251953125 0.8084701953125 0.1376953125 0.7460082109374999 0.08984375 0.6517259734375 0.04296875 0.559800790625 0.01171875 0.4961602796875 0.009765625 0.3900927609375 0.03515625 0.29816757968749996 0.076171875 0.2522049875 0.1064453125 0.15556569375 0.2080078125 0.049498175 0.3701171875 0.0188564484375 0.4638671875 0.0117852796875 0.5595703125 0.05185523125 0.7255859375 0.0942822390625 0.8017578125 0.14731599687500002 0.859375 0.2816681875 0.9453125 0.352379865625 0.974609375 0.4702326640625 0.990234375 0.5751216546875 0.9892578125 0.668225365625 0.974609375 0.7672217156250001 0.935546875 0.8450045625 0.88671875 0.9051094890625 0.8251953125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1081_jpg.rf.3fee39a6990fb23d44eea30539f59a86.txt
================================================
0 0.59663405625 0.8779296859375 0.617096234375 0.8925781265625 0.6117114484374999 0.8759765640625 0.655866678125 0.8535156265625 0.678482765625 0.8740234359375 0.6580205890624999 0.8925781265625 0.6407892796875 0.8769531265625 0.636481453125 0.8925781265625 0.66448233125 0.8964843734375 0.697867990625 0.8759765640625 0.69032929375 0.8671875 0.7150992984375 0.8720703140625 0.7150992984375 0.8525390640625 0.7301766921875 0.8486328140625 0.7194071265625 0.8115234359375 0.8357184578125001 0.6982421859375 0.8938741218749999 0.5576171859375 0.8960280375 0.4619140640625 0.8766428124999999 0.3916015640625 0.7926401875 0.1962890640625 0.6709440703125 0.1015625 0.599864925 0.078125 0.5137083828125 0.0859375 0.5094005562499999 0.05859375 0.49001533124999996 0.0859375 0.47493793906250004 0.05859375 0.4727840234375 0.078125 0.44478315 0.078125 0.440475321875 0.06445312656249999 0.43832140624999993 0.078125 0.31124050937500003 0.10546875 0.31877920625 0.11425781406249999 0.23262266406249998 0.1787109359375 0.178774821875 0.2470703140625 0.17015916874999998 0.2978515640625 0.1400043828125 0.3173828140625 0.152927859375 0.3349609359375 0.12277307500000001 0.3798828140625 0.11846524375 0.4248046859375 0.08615654218749999 0.4658203140625 0.11415742031250001 0.4873046859375 0.10123393437499999 0.4970703140625 0.11415742031250001 0.5302734359375 0.10338784999999999 0.6005859359375 0.1356965515625 0.6552734359375 0.1249269859375 0.6669921859375 0.152927859375 0.6865234359375 0.1432352515625 0.7207031265625 0.1378504671875 0.6962890640625 0.11846524375 0.6982421859375 0.12815785468750002 0.6875 0.10338784999999999 0.6865234359375 0.107695678125 0.6298828140625 0.08077175625 0.5683593734375 0.068925234375 0.5927734359375 0.12277307500000001 0.7412109359375 0.224007009375 0.8642578140625 0.362934434375 0.9433593734375 0.46309141093750006 0.9501953140625 0.533093603125 0.9433593734375 0.542786215625 0.9267578140625 0.528785775 0.9238281265625 0.5912492703125001 0.9160156265625 0.6052497109375 0.9013671859375 0.59663405625 0.8779296859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1099_jpg.rf.cd73f9b21c76fee7d8351df787a3a187.txt
================================================
0 0.8027343734375 0.6416015640625 0.8046875015625 0.4951171859375 0.720703125 0.2490234359375 0.6923828140625 0.1953124984375 0.6005859375 0.138671875 0.5400390625 0.1367187515625 0.4912109375 0.1601562484375 0.4462890625 0.1328124984375 0.3837890625 0.1445312484375 0.3115234359375 0.185546875 0.2617187515625 0.2724609375 0.1953124984375 0.4775390625 0.189453125 0.6259765640625 0.2246093734375 0.7080078140625 0.3486328140625 0.8027343734375 0.4248046890625 0.8378906265625 0.4902343734375 0.8388671859375 0.5654296890625 0.8359375015625 0.6865234359375 0.78125 0.7773437515625 0.6923828140625 0.8027343734375 0.6416015640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1111_jpg.rf.bcdda948159a5b6c6b9692d7153b122a.txt
================================================
0 0.6061112265624999 0.84375 0.7069027546875 0.8164062484375 0.7736430921874999 0.7666015625 0.8036081421875 0.7197265625 0.855365953125 0.6669921875 0.877158715625 0.6220703125 0.8826069078125001 0.4912109375 0.8145045234375001 0.2783203125 0.7177991375 0.1796875 0.660593134375 0.150390625 0.619731703125 0.146484375 0.5843184625 0.1328125 0.5352847453125 0.1328125 0.52166426875 0.1445312484375 0.49442331250000005 0.1328125 0.40452816562500005 0.138671875 0.3091848265625 0.166015625 0.2397203953125 0.2294921875 0.22609991875 0.2607421875 0.18796258125 0.2978515625 0.1525493421875 0.4072265625 0.119860196875 0.4580078125 0.119860196875 0.5595703125 0.1307565796875 0.6083984375 0.1906866796875 0.7236328125 0.2369963 0.7705078125 0.2982884453125 0.8125 0.382735403125 0.841796875 0.463096215625 0.8525390625 0.49987150468749997 0.841796875 0.5189401734375 0.818359375 0.5679738890625 0.84375 0.6061112265624999 0.84375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1130_jpg.rf.2db8f0176c51e5a76901704547e8d1c7.txt
================================================
0 0.7783865953125 0.6884765609375 0.8064727109375 0.6103515609375 0.8104850109375 0.5185546875 0.804466559375 0.4794921875 0.780392746875 0.4365234390625 0.7783865953125 0.3916015609375 0.7603312359375 0.3603515609375 0.7382635750000001 0.2333984390625 0.6981405546875 0.1357421875 0.6489898546874999 0.1015625 0.6028483796875 0.087890625 0.55269460625 0.056640625 0.5085592828125 0.060546875 0.4905039234375 0.083984375 0.4704424140625 0.064453125 0.44235629843749996 0.056640625 0.3681287125 0.08984374843750001 0.3420487484375 0.08984374843750001 0.2628057828125 0.1708984390625 0.22067661093750002 0.2666015609375 0.21465815781250003 0.3623046875 0.2006151015625 0.3740234390625 0.1805535921875 0.4482421875 0.1745351375 0.6240234390625 0.1966028015625 0.6552734390625 0.1925904984375 0.6708984390625 0.2006151015625 0.6923828125 0.22870121718749997 0.7236328125 0.2347196703125 0.7548828125 0.29590727656250004 0.8125 0.36010410937500004 0.84375 0.46241780937500004 0.84375 0.500534678125 0.8671875 0.5456730765625 0.8701171875 0.5667376625 0.8671875 0.6630329125000001 0.818359375 0.742275878125 0.7529296875 0.7783865953125 0.6884765609375


================================================
FILE: TumorDetection/valid/labels/no_tumor_117_jpg.rf.a77886592d46f8676d3f88224317d8bf.txt
================================================
0 0.900390625 0.48983651874999995 0.875 0.3199146390625 0.8447265609375 0.262225115625 0.6787109390625 0.1363570609375 0.5966796890625 0.1321614578125 0.5458984390625 0.15523726875 0.4267578109375 0.16362847187500001 0.38671875 0.1982421859375 0.4013671890625 0.205584490625 0.3681640609375 0.22236689843750002 0.3271484390625 0.205584490625 0.2421875 0.27376301875000003 0.22265625 0.32201244375 0.2373046890625 0.43004918906249995 0.3544921890625 0.48668981406249995 0.4638671890625 0.48668981406249995 0.48046875 0.5422815375 0.505859375 0.5653573515625 0.494140625 0.5779441578125 0.5048828109375 0.5706018515625 0.546875 0.63248698125 0.5634765609375 0.6964699078125001 0.6357421890625 0.727936921875 0.69921875 0.7289858234375 0.8330078109375 0.6419270828125 0.837890625 0.60311776875 0.892578125 0.5254991296875 0.900390625 0.48983651874999995


================================================
FILE: TumorDetection/valid/labels/no_tumor_1208_jpg.rf.98f94430f7427d8b40b37347d0a47511.txt
================================================
0 0.8754390046875 0.5185546890625 0.8195599171874999 0.4033203109375 0.78437679375 0.3779296890625 0.820594715625 0.3691406234375 0.8216295140625001 0.3447265625 0.7802376015625 0.3212890609375 0.7398804812499999 0.234375 0.6881405875 0.234375 0.6705490281250001 0.1845703109375 0.626052715625 0.1660156234375 0.5805216093750001 0.173828125 0.5184337390625 0.2460937484375 0.48221581249999995 0.2080078140625 0.4863550046875 0.1767578140625 0.45013707812500003 0.15625 0.41495395 0.1914062515625 0.36217925625 0.1923828140625 0.33010052343749996 0.2734374984375 0.3114741609375 0.2402343765625 0.28663901406250003 0.2382812515625 0.22351633906249999 0.3095703109375 0.2173075546875 0.4091796890625 0.1655676609375 0.4599609390625 0.1407325140625 0.5439453109375 0.14694129843749998 0.7216796890625 0.1841940234375 0.7939453109375 0.3011261828125 0.90625 0.38391001406249997 0.939453125 0.536025303125 0.9482421859375 0.6757230140625 0.9179687484375 0.8112815359375001 0.8017578140625 0.86302143125 0.6611328140625 0.8754390046875 0.5185546890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1213_jpg.rf.9c3c499dac7a19cc19fb3e7048a2b68d.txt
================================================
0 0.74609375 0.7309570328125 0.7558593765625 0.6723632828125 0.75390625 0.4731445328125 0.72265625 0.2416992171875 0.6376953125 0.087890625 0.6005859375 0.05859375 0.5302734375 0.038085935937499996 0.4287109375 0.052734375 0.3623046875 0.10546875 0.26953125 0.2651367171875 0.23828125 0.4614257828125 0.234375 0.6430664078125 0.2480468765625 0.6899414078125 0.2480468765625 0.7368164078125 0.2734375 0.8041992171875 0.3662109375 0.9287109359375 0.4169921875 0.9580078140625 0.4765625 0.9682617171875 0.5810546875 0.9580078140625 0.6376953125 0.9228515640625 0.7128906234375 0.8041992171875 0.74609375 0.7309570328125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1217_jpg.rf.51b55faf0f84152be6e3badb4a17dc26.txt
================================================
0 0.8652343765625 0.639758909375 0.8339843765625 0.3798253671875 0.7558593765625 0.1904164296875 0.6591796890625 0.0947044703125 0.5283203109375 0.062464651562499994 0.3857421890625 0.0906744890625 0.2832031234375 0.1763115109375 0.2519531234375 0.236761171875 0.2109375 0.4080352078125 0.19140625 0.575279271875 0.21484375 0.7646882078125 0.2832031234375 0.8896175062499999 0.3662109390625 0.9571196296875 0.4980468765625 0.9923819281249999 0.6572265609375 0.9571196296875 0.7509765609375 0.890625 0.84765625 0.7465533078125001 0.8652343765625 0.639758909375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1218_jpg.rf.346335503c22e7b3c561b3e653f9bf5c.txt
================================================
0 0.6972656265625 0.8544921859375 0.6796875015625 0.8466796890625 0.7402343734375 0.8037109375 0.814453125 0.6708984359375 0.8320312484375 0.5068359375 0.7539062484375 0.3310546890625 0.7441406265625 0.2451171859375 0.7109375015625 0.1982421859375 0.6298828140625 0.1328124984375 0.4853515640625 0.107421875 0.3251953109375 0.154296875 0.2617187515625 0.2294921859375 0.2226562484375 0.3154296890625 0.2226562484375 0.3583984359375 0.1953124984375 0.3623046890625 0.1933593734375 0.3935546890625 0.2041015640625 0.375 0.2128906265625 0.3837890625 0.1728515640625 0.5 0.1640624984375 0.4638671859375 0.1640624984375 0.6103515640625 0.1953124984375 0.7236328140625 0.2607421859375 0.8183593734375 0.2705078140625 0.810546875 0.3251953109375 0.8632812484375 0.3720703109375 0.876953125 0.4013671859375 0.9121093734375 0.5371093734375 0.9150390625 0.5888671859375 0.9140624984375 0.6650390625 0.859375 0.6767578140625 0.873046875 0.6972656265625 0.8544921859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1223_jpg.rf.7e04a1d8bdac4b0d82a00fb230faab4e.txt
================================================
0 0.944002890625 0.6142578109375 0.9303874625 0.4736328109375 0.83054100625 0.1923828109375 0.768136965625 0.09765624843750001 0.6024826125 0.019531248437500003 0.37328960625 0.0273437515625 0.215577584375 0.10644531406249999 0.1565774046875 0.2138671890625 0.0590001796875 0.5107421890625 0.0590001796875 0.6357421890625 0.0998464625 0.7470703140625 0.161115878125 0.8330078109375 0.312020184375 0.929687503125 0.52646315 0.9736328109375 0.6841751734375 0.9433593734375 0.815790959375 0.876953125 0.9076950859374999 0.7666015625 0.944002890625 0.6142578109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1224_jpg.rf.b32a1e11c44daa95b6dd7a247bce0915.txt
================================================
0 0.9075917125 0.4990234359375 0.8884341046875001 0.3857421875 0.75672554375 0.2021484359375 0.6405825390625 0.1328125 0.5088739828125 0.11328124843750001 0.3388502046875 0.140625 0.2203125 0.2158203125 0.1173403515625 0.4052734359375 0.09339334375 0.5439453125 0.1293138609375 0.6982421875 0.167629078125 0.7705078125 0.2286939515625 0.8320312484375 0.36279721249999997 0.8945312484375 0.5531759515625 0.9072265640625 0.755528190625 0.8574218765625 0.8333559781250001 0.7919921875 0.8908288046875 0.6767578125 0.9075917125 0.4990234359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1227_jpg.rf.20a2974c7b09a97daee67d10bbef77f8.txt
================================================
0 0.9550781234375 0.6845703125 0.96199898125 0.6162109390625 0.9273947031249999 0.3994140609375 0.902018228125 0.3232421875 0.7935914859375 0.1650390609375 0.734764209375 0.115234375 0.640179178125 0.068359375 0.474078634375 0.042968748437499996 0.347196275 0.0625 0.243383434375 0.103515625 0.12226845625 0.2099609390625 0.041525137499999996 0.3818359390625 0.0115347609375 0.6865234390625 0.1084267421875 0.8271484390625 0.2387695328125 0.9257812515625 0.36103798593750003 0.962890625 0.537519815625 0.9677734390625 0.697852978125 0.9335937484375 0.8039727687499999 0.8828125 0.9158599421875 0.7607421875 0.9550781234375 0.6845703125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1239_jpg.rf.e7f49a10e3c51a2f4716aced909a6879.txt
================================================
0 0.8134943187499999 0.8134765625 0.8582954546874999 0.7626953125 0.877159090625 0.7333984375 0.8960227265625 0.6708984375 0.912528409375 0.5576171875 0.9101704546875 0.4775390625 0.8748011359375001 0.3388671875 0.7969886359375 0.1923828125 0.7710511359375001 0.1611328125 0.7297869312499999 0.130859375 0.6331107968749999 0.09765625 0.5576562515625 0.08984375 0.46333806875 0.08984375 0.3831676140625 0.1015625 0.30771306875 0.12890625 0.26055397812500003 0.15625 0.2310795453125 0.1884765625 0.1933522734375 0.2607421875 0.16505681875 0.3369140625 0.15562499999999999 0.3955078125 0.15562499999999999 0.5458984375 0.18156250000000002 0.6826171875 0.2004261359375 0.7216796875 0.2098579546875 0.7626953125 0.2405113640625 0.8076171875 0.319502840625 0.865234375 0.41146306875 0.904296875 0.4987073859375 0.919921875 0.565909090625 0.9189453125 0.6566903421875 0.904296875 0.7203551125 0.87890625 0.7651562515625 0.853515625 0.8134943187499999 0.8134765625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1263_jpg.rf.278209afdaccffa333898c2a9806d076.txt
================================================
0 0.9485413640625 0.4228515625 0.9145027984375 0.2783203140625 0.8328102421875 0.1591796859375 0.715944496875 0.08398437343750001 0.5071746296875 0.0253906265625 0.3347125609375 0.0507812484375 0.189481346875 0.11914062656249999 0.08850027187500001 0.2373046859375 0.05219246875 0.3994140640625 0.0839617984375 0.5751953140625 0.15430816249999998 0.7802734375 0.215577584375 0.8876953140625 0.3596741796875 0.970703125 0.45384754374999997 0.9814453140625 0.6274442312499999 0.9726562484375 0.7851562484375 0.8935546859375 0.919041278125 0.5908203140625 0.9485413640625 0.4228515625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1284_jpg.rf.65e8ae14b72a58c15f86a06096c198c9.txt
================================================
0 0.798828125 0.5458984375 0.783203125 0.3798828125 0.720703125 0.2021484375 0.6318359375 0.12109375 0.5107421875 0.08984375 0.3896484375 0.103515625 0.29296875 0.1708984375 0.2421875 0.2587890625 0.19140625 0.4892578125 0.208984375 0.7021484375 0.267578125 0.8193359375 0.3876953125 0.90625 0.494140625 0.9169921875 0.6083984375 0.90234375 0.70703125 0.8369140625 0.779296875 0.7080078125 0.798828125 0.5458984375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1289_jpg.rf.88f9d17ca2ac7c8cc8d7b15fc82d832a.txt
================================================
0 0.9874218734375001 0.5830078125 0.9925781265625 0.2900390609375 0.9667968734375 0.2158203125 0.8108203140624999 0.07617187343750001 0.630351559375 0.0136718734375 0.4060546859375 0.011718746875 0.2126953140625 0.07421874687499999 0.06445312656249999 0.1904296875 0.023203126562500002 0.3427734390625 0.010312499999999999 0.5927734390625 0.08249999999999999 0.7666015609375 0.21785155937499998 0.9140625 0.318398440625 0.9667968734375 0.4408593734375 0.9873046875 0.6251953140625 0.9765625 0.8005078140625 0.8886718734375 0.9255468734375001 0.7314453125 0.9874218734375001 0.5830078125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1344_jpg.rf.6997999a2a135b672fd204796d5722cb.txt
================================================
0 0.798828125 0.5888671859375 0.8027343734375 0.3994140625 0.7773437515625 0.2861328140625 0.7099609375 0.1816406265625 0.6025390625 0.1328124984375 0.4638671859375 0.12109375156249999 0.3564453109375 0.154296875 0.2617187515625 0.2275390625 0.201171875 0.3876953109375 0.1992187515625 0.5263671859375 0.232421875 0.6728515640625 0.3085937515625 0.7998046890625 0.4208984359375 0.8710937515625 0.5429687515625 0.8759765640625 0.6123046890625 0.8632812484375 0.6630859375 0.8320312484375 0.7578124984375 0.7119140625 0.798828125 0.5888671859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1345_jpg.rf.4f408dfee96035c8070afc955b7dfc7d.txt
================================================
0 0.6669921859375 0.810546875 0.75 0.7314453109375 0.796875 0.6396484359375 0.810546875 0.5087890625 0.7578124984375 0.3857421859375 0.75 0.2783203109375 0.716796875 0.2236328140625 0.6318359375 0.15625 0.5341796890625 0.140625 0.5009765640625 0.154296875 0.4638671859375 0.138671875 0.3291015640625 0.1796875015625 0.25 0.2783203109375 0.2421875015625 0.3837890625 0.189453125 0.5029296890625 0.203125 0.6318359375 0.267578125 0.7470703109375 0.3701171859375 0.8359375015625 0.4453124984375 0.8544921859375 0.4755859375 0.8359375015625 0.4931640625 0.8515624984375 0.5693359375 0.8496093734375 0.6669921859375 0.810546875


================================================
FILE: TumorDetection/valid/labels/no_tumor_1348_jpg.rf.3560eb8c558f6b4596d9556129c45b03.txt
================================================
0 0.8183593734375 0.5322265640625 0.8085937515625 0.2587890625 0.7773437515625 0.1787109375 0.6630859375 0.06835937343750001 0.5302734359375 0.019531248437500003 0.4150390625 0.023437501562499997 0.3115234359375 0.07226562656249999 0.2148437515625 0.1689453109375 0.1679687515625 0.2802734359375 0.1660156265625 0.5576171859375 0.205078125 0.6865234359375 0.3232421859375 0.8359375015625 0.4609375015625 0.8798828140625 0.5888671859375 0.8710937515625 0.6904296890625 0.810546875 0.7773437515625 0.6767578140625 0.8183593734375 0.5322265640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1352_jpg.rf.03f11683c6e52a9c00f61c09aa474a26.txt
================================================
0 0.7871093734375 0.1923828140625 0.7216796890625 0.10156249843750001 0.6142578140625 0.046875 0.4287109375 0.035156248437499996 0.2958984359375 0.080078125 0.1816406265625 0.1962890625 0.158203125 0.3544921859375 0.169921875 0.5556640625 0.2148437515625 0.6904296890625 0.3408203109375 0.8515624984375 0.4130859375 0.8867187515625 0.544921875 0.8935546890625 0.6552734359375 0.8535156265625 0.8007812484375 0.6748046890625 0.8398437515625 0.3740234359375 0.7871093734375 0.1923828140625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1363_jpg.rf.483ea5869df59e48dc024e3271938801.txt
================================================
0 0.904307909375 0.4951171875 0.83257194375 0.2763671875 0.7717050671875 0.1826171875 0.6619273046874999 0.10546875156249999 0.5097601078125 0.08593750312500001 0.32281183593749996 0.13281249687500002 0.2217293453125 0.2158203140625 0.11303848593750002 0.4287109390625 0.09130031562499999 0.5927734359375 0.1608624625 0.7568359390625 0.26629259375 0.8574218734375 0.3815048984375 0.912109375 0.5391066390625 0.9189453140625 0.6880131078125 0.890625 0.7804003328125 0.8232421875 0.884743553125 0.6689453140625 0.904307909375 0.4951171875


================================================
FILE: TumorDetection/valid/labels/no_tumor_136_jpg.rf.78af9cb227bc4cf835b9e27085dc118e.txt
================================================
0 0.9633959109375001 0.4716796890625 0.9219333031250001 0.3076171875 0.8073013828125 0.1240234375 0.6573042984375 0.0507812515625 0.4621861390625 0.033203123437499996 0.29877468125 0.072265625 0.1902402046875 0.1513671875 0.1195098703125 0.2724609359375 0.0609744234375 0.4326171875 0.058535448437500005 0.6240234375 0.107314984375 0.7568359359375 0.1792648109375 0.8554687484375 0.364627059375 0.9492187484375 0.5658426609375 0.9677734375 0.754863378125 0.9199218765625 0.875592740625 0.8251953109375 0.94876205 0.6669921875 0.9633959109375001 0.4716796890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1384_jpg.rf.2b3633eb69760763da3edb53def51fcc.txt
================================================
0 0.830078125 0.7255859375 0.8320312484375 0.4306640625 0.7851562484375 0.2919921859375 0.6943359375 0.1796875015625 0.5693359375 0.11914062656249999 0.4345703109375 0.123046875 0.3681640625 0.1484375015625 0.2910156265625 0.2060546890625 0.2089843734375 0.3564453109375 0.185546875 0.4599609375 0.1933593734375 0.7373046890625 0.2382812484375 0.8388671859375 0.3369140625 0.9296875015625 0.4267578140625 0.9726562484375 0.560546875 0.9794921859375 0.6904296890625 0.9257812484375 0.7695312484375 0.8505859375 0.810546875 0.7802734359375 0.830078125 0.7255859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1386_jpg.rf.04f978ef6b137d199b27baeb69b47fd6.txt
================================================
0 0.7910156265625 0.5029296890625 0.7578124984375 0.3916015640625 0.7539062484375 0.2900390625 0.6816406265625 0.1806640625 0.5673828140625 0.125 0.5283203109375 0.125 0.5009765640625 0.15625 0.4892578140625 0.1328124984375 0.4384765640625 0.11523437343750001 0.3408203109375 0.1484375015625 0.2578124984375 0.2744140625 0.2109375015625 0.4599609375 0.2109375015625 0.5986328140625 0.236328125 0.6943359375 0.3076171859375 0.8203124984375 0.4150390625 0.8964843734375 0.5351562484375 0.9130859375 0.6279296890625 0.8808593734375 0.732421875 0.7666015640625 0.7871093734375 0.6435546890625 0.7910156265625 0.5029296890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1396_jpg.rf.164ed037bd3b4078bf95bfebfbe2f577.txt
================================================
0 0.9417336531249999 0.5068359359375 0.8396179531250001 0.1982421890625 0.7851562484375 0.10644531406249999 0.6251749890625 0.0273437515625 0.40278969375000007 0.019531248437500003 0.241673815625 0.08593750312500001 0.16565435312500001 0.1943359359375 0.0658078953125 0.4794921890625 0.05219246875 0.6201171890625 0.0839617984375 0.7607421890625 0.1781351625 0.875 0.323366375 0.9492187515625 0.5060400109375001 0.9775390640625 0.7000598375 0.9257812484375 0.8396179531250001 0.8349609359375 0.9326567000000001 0.6650390640625 0.9417336531249999 0.5068359359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1410_jpg.rf.4dc51a0a8672ccb40ab19e2ac0d45d24.txt
================================================
0 0.95955768125 0.4384765640625 0.8453824624999999 0.2041015640625 0.777363184375 0.1142578125 0.6401099953125 0.037109376562500004 0.5332225578124999 0.017578123437500003 0.4481984625 0.0234375 0.2927258265625 0.078125 0.1651896765625 0.2119140640625 0.058302239062500004 0.4306640640625 0.051014457812500004 0.6533203125 0.145755596875 0.8349609359375 0.2538576671875 0.921875 0.38989622343749997 0.9570312484375 0.6437538859374999 0.9560546875 0.7834363328125 0.9121093765625 0.8939676609375 0.8037109359375 0.9474113828125 0.6728515640625 0.95955768125 0.4384765640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1436_jpg.rf.f9dfa062fa7fe2faad9bf6f2339bc801.txt
================================================
0 0.841796875 0.4638671859375 0.765625 0.2451171859375 0.6826171859375 0.1347656265625 0.5595703109375 0.08593750156249999 0.4365234359375 0.08398437343750001 0.3134765640625 0.1347656265625 0.236328125 0.2294921859375 0.1640624984375 0.4306640625 0.154296875 0.5908203109375 0.1972656265625 0.7412109375 0.2919921859375 0.859375 0.4150390625 0.921875 0.5507812484375 0.9248046890625 0.6689453109375 0.888671875 0.7890624984375 0.7626953109375 0.841796875 0.5927734359375 0.841796875 0.4638671859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1446_jpg.rf.2bd824b13bed9b4682994b744ea84bd3.txt
================================================
0 0.88310455625 0.3857421859375 0.8012558406249999 0.2451171859375 0.69032929375 0.1503906265625 0.599864925 0.12304687343750001 0.5007849015625 0.140625 0.36939617343750003 0.1386718734375 0.2358535328125 0.21875 0.1636974296875 0.3037109359375 0.11415742031250001 0.4033203140625 0.11415742031250001 0.5556640640625 0.1809287390625 0.7568359359375 0.2358535328125 0.8359375 0.3672422609375 0.9140625 0.5557096968749999 0.9189453140625 0.694637121875 0.87890625 0.7969480140625 0.7861328140625 0.868027159375 0.6181640640625 0.88310455625 0.3857421859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1449_jpg.rf.23f0a67b927372d9d62fa82ae384d29d.txt
================================================
0 0.76171875 0.5375976578125 0.765625 0.3618164078125 0.75390625 0.3237304671875 0.75390625 0.2739257828125 0.73046875 0.2036132828125 0.6376953125 0.076171875 0.5615234375 0.03515625 0.4248046875 0.041015625 0.3583984375 0.08203125 0.2578125 0.2592773421875 0.2441406234375 0.3237304671875 0.2480468765625 0.5551757828125 0.2734375 0.7485351578125 0.3642578125 0.9140625 0.4287109375 0.955078125 0.5253906234375 0.9624023421875 0.5927734375 0.9375 0.6279296875 0.908203125 0.73046875 0.7338867171875 0.76171875 0.5375976578125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1471_jpg.rf.70f8826cb6dd5840e3f3e8cdab81914e.txt
================================================
0 0.857421875 0.4462890625 0.826171875 0.2744140625 0.7666015640625 0.1777343734375 0.6005859375 0.08789062656249999 0.4150390625 0.08398437343750001 0.3466796890625 0.10546875156249999 0.2451171859375 0.1757812484375 0.1835937515625 0.2607421859375 0.1503906265625 0.4150390625 0.1503906265625 0.6025390625 0.185546875 0.7099609375 0.2802734359375 0.8496093734375 0.4130859375 0.9179687515625 0.5546875015625 0.9228515640625 0.6904296890625 0.8789062484375 0.8125 0.7412109375 0.8496093734375 0.6513671859375 0.857421875 0.4462890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1498_jpg.rf.0cc5de5358026d422cca78938a1b524c.txt
================================================
0 0.9675611406250001 0.4326171890625 0.94004755625 0.2900390625 0.831139603125 0.1367187484375 0.7577700390625 0.08984374843750001 0.6316661015625 0.044921876562500004 0.47117017812500006 0.042968748437499996 0.33818784062499996 0.087890625 0.21552309687500001 0.1748046875 0.1329823375 0.3251953125 0.10546874843750001 0.5556640625 0.14215353125000002 0.6787109375 0.2040591015625 0.7861328109375 0.29921025625 0.8789062515625 0.388629415625 0.9335937484375 0.4505349859375 0.9511718765625 0.6213485078125001 0.9541015640625 0.7302564546875 0.9179687484375 0.8483355984375001 0.8154296875 0.9583899468749999 0.5888671890625 0.9675611406250001 0.4326171890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1516_jpg.rf.d295bf54bf757e4c11e3227514b24927.txt
================================================
0 0.94482421875 0.5888671890625 0.9350585953125 0.4169921890625 0.8984375 0.3076171890625 0.8227539046875 0.1689453125 0.706787109375 0.07421874843750001 0.5358886734375 0.03125 0.3796386734375 0.041015625 0.2673339859375 0.078125 0.1953125 0.12011718906249999 0.08056640468750001 0.3037109375 0.036621095312500004 0.4794921890625 0.04638671875 0.6708984359375 0.13671875 0.8427734359375 0.2404785140625 0.9199218765625 0.328369140625 0.9511718765625 0.5029296859375 0.9697265640625 0.6457519546875 0.9492187484375 0.8300781265625 0.8505859375 0.9033203140625 0.7373046875 0.94482421875 0.5888671890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1526_jpg.rf.f04364a9a0677f4c5afa9c6722332652.txt
================================================
0 0.69921875 0.6817756312500001 0.75390625 0.50637240625 0.744140625 0.33514544687500003 0.716796875 0.264148903125 0.6591796890625 0.17540322656250001 0.5380859390625 0.0939660140625 0.4873046890625 0.1127592171875 0.4794921890625 0.1002304140625 0.4033203109375 0.106494815625 0.3173828109375 0.150345621875 0.228515625 0.2620607734375 0.1953125 0.383172525 0.205078125 0.5523113453125 0.28125 0.7736535109375 0.3935546890625 0.845694125 0.4365234390625 0.851958525 0.4697265609375 0.83525345625 0.51171875 0.8509144609374999 0.5869140609375 0.8039314515625 0.6181640609375 0.8039314515625 0.654296875 0.75903658125 0.69921875 0.6817756312500001


================================================
FILE: TumorDetection/valid/labels/no_tumor_1527_jpg.rf.48e4573bae8fb915b51adb01bbb9ee74.txt
================================================
0 0.904307909375 0.4931640609375 0.83257194375 0.2744140609375 0.7608359828125 0.1669921875 0.6401891359375 0.099609375 0.48584811875 0.087890625 0.32281183593749996 0.13281249687500002 0.219555525 0.2177734359375 0.11303848593750002 0.4287109390625 0.09130031562499999 0.5908203140625 0.15651483125 0.7490234359375 0.26629259375 0.8574218734375 0.39020016874999996 0.914062496875 0.5391066390625 0.9189453140625 0.6923607453125 0.8886718734375 0.7977908703125001 0.8056640609375 0.884743553125 0.6669921875 0.904307909375 0.4931640609375


================================================
FILE: TumorDetection/valid/labels/no_tumor_153_jpg.rf.c20634846bf0ce46b0448de73a93b798.txt
================================================
0 0.899864784375 0.8427734375 0.9793982890624999 0.6435546859375 0.9725811296875 0.3994140625 0.886230471875 0.2177734375 0.723754884375 0.0703125 0.5874117359375 0.017578123437500003 0.3965313265625 0.017578123437500003 0.2397367015625 0.07421875156249999 0.12043644687499999 0.1923828140625 0.020451471875 0.4013671859375 0.020451471875 0.6396484375 0.0931678171875 0.8173828140625 0.2101956875 0.9199218765625 0.378352240625 0.9804687515625 0.4624305140625 0.9863281234375 0.4942439140625 0.96875 0.49651630312500006 0.9824218765625 0.5544621390625 0.9853515625 0.6487661484375 0.9765625 0.7737473703125 0.9355468765625 0.8510084906250001 0.8984375 0.899864784375 0.8427734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_1540_jpg.rf.f77cfa5a2c91216f53de79005f23a689.txt
================================================
0 0.8545421484375 0.6142578125 0.8545421484375 0.4404296875 0.8035247062499999 0.2744140640625 0.7567587203125 0.1962890640625 0.69192405625 0.1367187515625 0.5728833578125 0.08203124843750001 0.3836936796875 0.08984375156249999 0.31354469375 0.1171875 0.22320130625 0.1845703125 0.14029796249999998 0.3251953125 0.0892805203125 0.5283203125 0.0977834328125 0.6591796875 0.174309590625 0.8056640640625 0.2433957109375 0.8632812484375 0.33055050625 0.90625 0.48891715312499995 0.9208984359375 0.6175236187500001 0.9023437515625 0.70680414375 0.8535156234375 0.8035247062499999 0.7392578125 0.8545421484375 0.6142578125


================================================
FILE: TumorDetection/valid/labels/no_tumor_1543_jpg.rf.f3ad98a16dad21108457852f884f3620.txt
================================================
0 0.6455078140625 0.8671875015625 0.736328125 0.7392578140625 0.7617187515625 0.6259765640625 0.7578124984375 0.5166015640625 0.7109375015625 0.4052734359375 0.7109375015625 0.3212890625 0.6357421859375 0.2128906265625 0.5380859375 0.1816406265625 0.4287109375 0.1816406265625 0.3183593734375 0.2470703109375 0.2714843734375 0.3291015640625 0.28125 0.3955078140625 0.2402343734375 0.4658203109375 0.216796875 0.5986328140625 0.2246093734375 0.7041015640625 0.2753906265625 0.8017578140625 0.3916015640625 0.9121093734375 0.4589843734375 0.9267578140625 0.4931640625 0.8984375015625 0.5244140625 0.919921875 0.5673828140625 0.9179687515625 0.6455078140625 0.8671875015625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1560_jpg.rf.c6bdac2fa89008814012673fe60d3508.txt
================================================
0 0.830078125 0.6298828140625 0.841796875 0.4560546890625 0.798828125 0.2958984359375 0.7275390625 0.1679687515625 0.5771484359375 0.07421875156249999 0.4267578140625 0.06640624843750001 0.2861328140625 0.11914062656249999 0.2128906265625 0.2177734359375 0.1523437515625 0.4619140625 0.1601562484375 0.5849609375 0.203125 0.7236328140625 0.3056640625 0.8535156265625 0.4248046890625 0.9179687515625 0.5390624984375 0.9248046890625 0.6201171859375 0.9023437515625 0.6865234359375 0.861328125 0.767578125 0.7705078140625 0.830078125 0.6298828140625


================================================
FILE: TumorDetection/valid/labels/no_tumor_1566_jpg.rf.f57ac5bf9b8a5026a8f5ff3726afe00a.txt
================================================
0 0.8524164234375 0.5576171875 0.8524164234375 0.3837890640625 0.767387353125 0.2099609359375 0.664289609375 0.12304687656249999 0.5473746375 0.08398437656249999 0.366687865625 0.08789062343750001 0.217886990625 0.1601562484375 0.1636809578125 0.2099609359375 0.12754360468749998 0.2744140640625 0.0892805203125 0.4013671875 0.114789246875 0.6005859359375 0.18068677187499999 0.7529296875 0.2689044328125 0.8515625 0.3879451296875 0.9101562484375 0.5633175859375 0.9169921875 0.6834211484375 0.8691406234375 0.7546329953125 0.8037109359375 0.8035247062499999 0.7216796875 0.8524164234375 0.5576171875


================================================
FILE: TumorDetection/valid/labels/no_tumor_1585_jpg.rf.8f7ba4467c93dde0aab4f224746fdbb5.txt
================================================
0 0.8076171859375 0.2539062484375 0.8046875015625 0.2138671859375 0.7021484359375 0.1289062484375 0.6728515640625 0.140625 0.6669921859375 0.111328125 0.5927734359375 0.08203124843750001 0.4677734359375 0.10351562656249999 0.4443359375 0.123046875 0.3720703109375 0.11523437343750001 0.3505859375 0.1484375015625 0.3212890625 0.1484375015625 0.2783203109375 0.1914062484375 0.220703125 0.2158203109375 0.2281366625 0.2397728171875 0.2138671859375 0.232421875 0.1962890625 0.2382812484375 0.080078125 0.3779296890625 0.07226562656249999 0.4345703109375 0.12011718593750001 0.515625 0.1826171859375 0.5234375015625 0.2509765640625 0.5585937515625 0.3623046890625 0.5683593734375 0.4013671859375 0.501953125 0.4013671859375 0.5546875015625 0.4716796890625 0.5351562484375 0.5742187515625 0.4189453109375 0.546875 0.3642578140625 0.470703125 0.2998046890625 0.5029296890625 0.2910156265625 0.5751953109375 0.3164062484375 0.5996093734375 0.3408203109375 0.580078125 0.3701171859375 0.5859375015625 0.4521484359375 0.5566406265625 0.5263671859375 0.5146484359375 0.517578125 0.482421875 0.5673828140625 0.5810546890625 0.6191406265625 0.6054687515625 0.6767578140625 0.6484375015625 0.6982421859375 0.814453125 0.5966796890625 0.9140624984375 0.4130859375 0.8632812484375 0.2763671859375 0.8310546890625 0.248046875 0.8076171859375 0.2539062484375


================================================
FILE: TumorDetection/valid/labels/no_tumor_160_jpg.rf.7625f9c90ed889d0c404d10336754e20.txt
================================================
0 0.9755907953125 0.6533203109375 0.94388409375 0.4052734375 0.86583683125 0.2314453109375 0.7841311046875 0.1523437484375 0.6012078265625 0.07421874843750001 0.3963337609375 0.07031250156249999 0.1890207171875 0.150390625 0.10975396562499999 0.2138671875 0.07073033281249999 0.2763671875 0.009755910937499999 0.4716796890625 0.014633860937500002 0.7548828125 0.092681125 0.8798828125 0.1817037859375 0.9550781234375 0.2499951390625 0.9882812515625 0.30121365624999996 1 0.6560848109374999 0.9990234375 0.8353496171874999 0.921875 0.8951045578124999 0.8701171875 0.9316892093750001 0.8095703109375 0.9755907953125 0.6533203109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_163_jpg.rf.b26bfb07fbe3a327ceee190d2a903c1b.txt
================================================
0 0.8710937484375 0.6760635734375 0.8828125015625 0.44699255937500004 0.849609375 0.3213075734375 0.755859375 0.1672421140625 0.6435546890625 0.060815312499999996 0.4912109359375 0.0162174171875 0.4169921875 0.028380478125000004 0.3564453109375 0.0628424921875 0.212890625 0.211840009375 0.134765625 0.384150065625 0.12304687656249999 0.5463242375 0.1328125015625 0.6416015640625 0.2207031234375 0.8321562124999999 0.3779296890625 0.9791265546875 0.5449218765625 0.994330384375 0.5810546890625 0.9933167968750001 0.7275390640625 0.9142568859375 0.7929687484375 0.84026491875 0.8710937484375 0.6760635734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_171_jpg.rf.c1116241df77b03e6d4ab77a910b4ac0.txt
================================================
0 0.7128906265625 0.7392578140625 0.7734375 0.6279296859375 0.7832031265625 0.4775390640625 0.7402343734375 0.3232421859375 0.6572265640625 0.2324218734375 0.5341796859375 0.1777343734375 0.3857421859375 0.1953125 0.2792968734375 0.2744140640625 0.2402343734375 0.3505859359375 0.2089843734375 0.5048828140625 0.203125 0.6005859359375 0.2265625 0.6806640640625 0.2890625 0.7802734359375 0.3486328140625 0.8378906265625 0.421875 0.8603515640625 0.4951171859375 0.8378906265625 0.5556640640625 0.8574218734375 0.6025390640625 0.8515625 0.7128906265625 0.7392578140625


================================================
FILE: TumorDetection/valid/labels/no_tumor_183_jpg.rf.aaa5d2efc50933db68c8052329d6fd4b.txt
================================================
0 0.8921201890624999 0.5771484375 0.8921201890624999 0.5419921875 0.8603460453125 0.4775390625 0.8554577140624999 0.4365234375 0.8016860875 0.3505859375 0.7845769328125 0.2939453109375 0.728361140625 0.2373046890625 0.639149121875 0.193359375 0.5511591859375 0.177734375 0.52182920625 0.177734375 0.49494339218750005 0.19140625 0.47294590937500003 0.177734375 0.44117176562499993 0.17578125 0.39228846562499997 0.203125 0.32385185 0.2109374984375 0.2444164921875 0.2568359375 0.23463983124999999 0.2939453109375 0.205309853125 0.3095703109375 0.1833123671875 0.3779296890625 0.136873234375 0.4384765625 0.1344290703125 0.4833984375 0.1197640796875 0.5146484375 0.1197640796875 0.6005859375 0.1637590484375 0.7099609375 0.2651918921875 0.83203125 0.3067426953125 0.853515625 0.3507376640625 0.8613281234375 0.40206512656250004 0.8984374984375 0.5267175374999999 0.89453125 0.5841554125 0.9013671875 0.6611466046875 0.884765625 0.7625794484374999 0.8212890625 0.8481252203125 0.7099609375 0.8921201890624999 0.5771484375


================================================
FILE: TumorDetection/valid/labels/no_tumor_185_jpg.rf.e9332674f3901590135928591ee7b443.txt
================================================
0 0.7177734375 0.7531778984375 0.720703125 0.7445007328125001 0.71484375 0.6750834140624999 0.7041015625 0.66640625 0.6904296875 0.6768188484375 0.67578125 0.6611999515625 0.67578125 0.6369038890625001 0.701171875 0.60219523125 0.70703125 0.5778991703125 0.69921875 0.34535115625 0.685546875 0.26899210625 0.6357421875 0.18048502500000002 0.6044921875 0.1457763671875 0.5654296875 0.11800943906250001 0.5419921875 0.1110677078125 0.4794921875 0.11800943906250001 0.4306640625 0.1527181 0.3857421875 0.2082519546875 0.3671875 0.2446960453125 0.34765625 0.3037007640625 0.333984375 0.3765889484375 0.32421875 0.5293070484375 0.33984375 0.5813700359375 0.333984375 0.6299621578125 0.322265625 0.6473164875 0.3125 0.699379475 0.326171875 0.7236755375 0.330078125 0.7792093921875 0.34375 0.8347432453125 0.3515625 0.9076314296874999 0.3564453125 0.91630859375 0.3896484375 0.9093668625 0.4130859375 0.91630859375 0.5244140625 0.91630859375 0.5556640625 0.9093668625 0.66796875 0.9145731609375 0.6953125 0.7479715984374999 0.7001953125 0.7427653000000001 0.7080078125 0.7531778984375 0.7177734375 0.7531778984375


================================================
FILE: TumorDetection/valid/labels/no_tumor_194_jpg.rf.a826cac67a65f27a35b74ea688103344.txt
================================================
0 0.9243284624999999 0.9423828125 0.926359953125 0.8857421875 0.9202654796875 0.8798828125 0.9243284624999999 0.8701171875 0.9182339890625 0.8701171875 0.9162024984375 0.8427734375 0.9182339890625 0.7041015625 0.9243284624999999 0.6767578125 0.9162024984375 0.6689453125 0.9162024984375 0.6474609375 0.9548008296875 0.5595703125 0.97308425 0.4853515625 0.9791787218750001 0.3837890625 0.97308425 0.3642578125 0.9710527593749999 0.3076171875 0.9243284624999999 0.1845703125 0.8776041671874999 0.1201171875 0.8075177218749999 0.05859375 0.77704535625 0.0390625 0.6612503609375 0.0078125 0.55358133125 0.005859375 0.484510634375 0.017578125 0.41543993593749995 0.037109375 0.28745599374999997 0.1015625 0.205180603125 0.1669921875 0.1482988515625 0.2451171875 0.12798394062500001 0.3095703125 0.11782648437500001 0.4013671875 0.12392095781250001 0.4384765625 0.138141396875 0.4716796875 0.138141396875 0.5185546875 0.1218894671875 0.5517578125 0.058913242187500006 0.6259765625 0.046724296875 0.6552734375 0.050787278125 0.6845703125 0.089385609375 0.7216796875 0.0975115734375 0.7412109375 0.0975115734375 0.8115234375 0.11376350312500001 0.8564453125 0.13306266875 0.873046875 0.14931459687499998 0.875 0.1686137640625 0.8876953125 0.176739728125 0.9541015625 0.2143223125 0.98046875 0.2204167875 0.9921875 0.48552637968749995 0.9990234375 0.880651403125 0.9921875 0.9212812250000001 0.99609375 0.932454428125 0.9794921875 0.9243284624999999 0.9423828125


================================================
FILE: TumorDetection/valid/labels/no_tumor_197_jpg.rf.d86f7d9652a73b11fa6a21d1c9cd60c5.txt
================================================
0 0.7075545484375 0.802734375 0.75645465 0.7548828125 0.8063527125000001 0.6396484359375 0.8043567906250001 0.5244140625 0.7664342625 0.3544921875 0.7105484296875 0.2509765640625 0.6826055156249999 0.2275390625 0.6726259031249999 0.2021484359375 0.6237258015625 0.1640624984375 0.5937869625000001 0.1640624984375 0.5638481234375 0.15234375 0.5019745234374999 0.154296875 0.484011221875 0.162109375 0.43411315781250004 0.15625 0.41215800781249995 0.166015625 0.3742354796875 0.16796875 0.33431702812499997 0.185546875 0.3073720734375 0.2099609375 0.277433234375 0.2548828125 0.23352294062499998 0.3857421875 0.2135637125 0.5419921875 0.2155596375 0.5908203125 0.2255392484375 0.6572265640625 0.24749439843749999 0.7119140625 0.2754373125 0.7626953125 0.314357803125 0.8046875015625 0.4001824734375 0.857421875 0.43411315781250004 0.8671875015625 0.49598675625000005 0.861328125 0.5119541375 0.873046875 0.52692355625 0.8720703125 0.5418929734375 0.859375 0.5957828843749999 0.8671875015625 0.63969318125 0.849609375 0.7075545484375 0.802734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_206_jpg.rf.c3e7e854c21aa7a726c844abaacfa985.txt
================================================
0 0.5498076375000001 0.931640625 0.6049081843749999 0.935546875 0.664800084375 0.908203125 0.753440096875 0.890625 0.8133319968749999 0.859375 0.85286065 0.8271484390625 0.92952228125 0.7255859390625 0.963061746875 0.6337890609375 0.9654574203124999 0.5849609390625 0.93431363125 0.4306640609375 0.862443353125 0.3154296875 0.84088226875 0.2177734390625 0.8001557765625 0.1787109390625 0.773803340625 0.1298828125 0.7342746890625 0.1015625 0.6504260296875 0.072265625 0.528246553125 0.056640625 0.4803330328125 0.076171875 0.4467935703125 0.052734375 0.4156497828125 0.0507812515625 0.358153559375 0.056640625 0.2168086765625 0.109375 0.16530164375 0.1591796875 0.11499244687500002 0.2646484390625 0.12457515156250001 0.3154296875 0.08145298437500001 0.3779296875 0.07666163125 0.4208984390625 0.055100546875 0.4677734390625 0.055100546875 0.6025390609375 0.10540974375 0.7412109390625 0.1629059671875 0.8173828125 0.27909625 0.8984375 0.3605492359375 0.9140625 0.3940886984375 0.9375 0.4575741125 0.9443359390625 0.4863222234375 0.9365234390625 0.4887179 0.9169921875 0.5138724984375 0.8945312515625 0.5498076375000001 0.931640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_216_jpg.rf.81d70848b85647aaacd54e6327c689f9.txt
================================================
0 0.9242497265624999 0.6611328125 0.9332667921875 0.5146484390625 0.8701473046875 0.2998046875 0.7247470390624999 0.11914062656249999 0.5579312375 0.0585937484375 0.4046410390625 0.0605468734375 0.23106243124999998 0.125 0.139764590625 0.2255859390625 0.0743908328125 0.3681640609375 0.045085353125 0.5126953125 0.045085353125 0.6064453125 0.0946792421875 0.7802734390625 0.176960009375 0.859375 0.3302502078125 0.9335937484375 0.5117187484375 0.9462890609375 0.7089671671875 0.9101562515625 0.8588759640625 0.8154296875 0.894944246875 0.7705078125 0.9242497265624999 0.6611328125


================================================
FILE: TumorDetection/valid/labels/no_tumor_217_jpg.rf.8f847ed9bd96be84f764ece15d91b6e6.txt
================================================
0 0.6513671875 0.921875 0.6669921875 0.91796875 0.7099609375 0.888671875 0.76171875 0.8388671875 0.80859375 0.7841796875 0.822265625 0.7626953125 0.83203125 0.7373046875 0.84765625 0.6767578125 0.857421875 0.5615234375 0.84765625 0.4892578125 0.826171875 0.4130859375 0.80078125 0.2958984375 0.76171875 0.2001953125 0.740234375 0.1611328125 0.6748046875 0.08984375 0.6513671875 0.07421875 0.5830078125 0.048828125 0.4931640625 0.03515625 0.4267578125 0.04296875 0.3779296875 0.0546875 0.3564453125 0.064453125 0.3095703125 0.09765625 0.26953125 0.1376953125 0.24609375 0.1669921875 0.21875 0.2216796875 0.19140625 0.3017578125 0.14453125 0.4892578125 0.14453125 0.5185546875 0.138671875 0.5517578125 0.138671875 0.5810546875 0.16796875 0.7255859375 0.19140625 0.7783203125 0.24609375 0.8486328125 0.3017578125 0.90234375 0.3564453125 0.9375 0.3857421875 0.94921875 0.4365234375 0.9609375 0.484375 0.9599609375 0.5537109375 0.95703125 0.5908203125 0.94921875 0.6357421875 0.93359375 0.6513671875 0.921875


================================================
FILE: TumorDetection/valid/labels/no_tumor_236_jpg.rf.85c9441997ad78e26443f19c05332dd9.txt
================================================
0 0.5322265625 0.8611752718749999 0.5380859375 0.8675543484375 0.5771484375 0.8675543484375 0.6689453125 0.8250271734375 0.7060546875 0.7995108703125 0.7294921875 0.7676154890625 0.7646484375 0.74422554375 0.8046875 0.660234375 0.828125 0.5475373640625001 0.818359375 0.50713655 0.796875 0.47311481093749996 0.796875 0.4178294828125 0.7695312484375 0.34553328906249997 0.767578125 0.29024796249999996 0.7148437515625 0.211572690625 0.705078125 0.18393002812499998 0.6787109375 0.148845109375 0.6474609375 0.127581521875 0.5810546875 0.0978125015625 0.5439453125 0.095686140625 0.5439453125 0.0893070640625 0.5283203125 0.08718070624999999 0.5185546875 0.091433425 0.5048828125 0.0829279890625 0.4658203125 0.0893070640625 0.4521484375 0.10631793437499999 0.4150390625 0.11057065156250001 0.3935546875 0.131834240625 0.3876953125 0.12545516406249999 0.3466796875 0.144592390625 0.3320312484375 0.1732982328125 0.310546875 0.19668817968750002 0.302734375 0.22007812499999999 0.275390625 0.2413417109375 0.2695312484375 0.271110734375 0.275390625 0.326396059375 0.244140625 0.38806046250000004 0.2460937515625 0.39869225625000004 0.2265625 0.439093071875 0.2265625 0.45823029843749996 0.205078125 0.48799932031250004 0.1953125 0.5730536671875 0.234375 0.6921297546875 0.259765625 0.7367832890625 0.2919921875 0.773994565625 0.3662109375 0.8356589671875 0.4462890625 0.8675543484375 0.4873046875 0.8590489124999999 0.4931640625 0.8718070640625 0.513671875 0.874996603125 0.5224609375 0.873933425 0.5234375 0.8643648109375001 0.5302734375 0.8675543484375 0.5322265625 0.8611752718749999


================================================
FILE: TumorDetection/valid/labels/no_tumor_302_jpg.rf.de592ff09d80dfcb17d95a8ede53d1e4.txt
================================================
0 0.9710365859375001 0.7158203125 0.9943788093749999 0.6494140625 0.9920445875 0.3349609375 0.9710365859375001 0.2626953125 0.9196836875000001 0.1708984375 0.8729992390625 0.1220703125 0.8018054484375 0.072265625 0.7364472171875 0.042968751562500004 0.6547494296874999 0.0273437515625 0.47268006875 0.025390625 0.3629716078125 0.048828125 0.29527915312499997 0.08203124843750001 0.1984089171875 0.1513671875 0.14939024375 0.2060546875 0.1050400140625 0.2783203125 0.0256764484375 0.4501953125 0.0116711125 0.5908203125 0.01867378125 0.6552734375 0.0583555640625 0.7490234375 0.11437690468749999 0.8408203125 0.16572980312500002 0.8955078125 0.24859470312499998 0.958984375 0.36764005312500003 1 0.6629192078125 0.9990234375 0.7667921125 0.9492187515625 0.8123094515625 0.9130859375 0.9150152437499999 0.8095703125 0.9710365859375001 0.7158203125


================================================
FILE: TumorDetection/valid/labels/no_tumor_329_jpg.rf.070fe5b5666fe2889a0516d4e93f09db.txt
================================================
0 0.9456104328125001 0.4384765625 0.91681011875 0.3662109375 0.8304091609375 0.2587890625 0.6804075078125 0.1367187515625 0.586806475 0.10742187656249999 0.3804041984375 0.11328124843750001 0.28680316562499997 0.1601562484375 0.14880164375 0.3017578140625 0.072000796875 0.4775390625 0.072000796875 0.6005859375 0.1296014328125 0.7236328140625 0.22680250468750002 0.8105468765625 0.3996044109375 0.8710937515625 0.56640625 0.8779296859375 0.7332080890625 0.8457031234375 0.8784096953124999 0.7607421859375 0.9504104874999999 0.5869140625 0.9456104328125001 0.4384765625


================================================
FILE: TumorDetection/valid/labels/no_tumor_32_jpg.rf.0fb9b63cf476fafec16401c5822419ea.txt
================================================
0 0.7109375 0.4775390640625 0.6464843734375 0.3447265640625 0.5693359359375 0.2734375 0.4853515640625 0.2558593734375 0.4248046859375 0.2890625 0.3974609359375 0.26953125 0.3525390640625 0.2753906265625 0.2578125 0.3388671859375 0.21875 0.4189453140625 0.1933593734375 0.5478515640625 0.203125 0.7333984359375 0.2792968734375 0.8486328140625 0.3681640640625 0.9121093734375 0.4365234359375 0.9355468734375 0.5488281265625 0.9365234359375 0.6611328140625 0.8671875 0.7109375 0.7880859359375 0.73828125 0.6728515640625 0.7109375 0.4775390640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_355_jpg.rf.84477c90ae6a91a6d7d46575f37742c5.txt
================================================
0 0.7952860171875 0.6337890625 0.7930266828125 0.5087890625 0.763655321875 0.4912109375 0.777211334375 0.4775390625 0.7749520000000001 0.4345703109375 0.7523586484375 0.4130859375 0.763655321875 0.3837890625 0.67441158125 0.283203125 0.648429225 0.2861328125 0.5885568390625 0.2089843734375 0.5659634875 0.2109375015625 0.5478888046875 0.234375 0.5241657843749999 0.1748046890625 0.49592409374999996 0.1660156265625 0.472201071875 0.1962890625 0.45751539374999994 0.2734375015625 0.42475503437500006 0.1845703109375 0.4010320125 0.1777343734375 0.376179321875 0.2089843734375 0.32421461093750004 0.205078125 0.3129179359375 0.2421875015625 0.2812872421875 0.2382812484375 0.2055995078125 0.3056640625 0.18300615625 0.3564453109375 0.19204349999999998 0.3857421875 0.1468567921875 0.4345703109375 0.1333007796875 0.5146484359375 0.1468567921875 0.5556640625 0.137819453125 0.5908203109375 0.164931475 0.6689453109375 0.2440082125 0.7802734359375 0.3490673015625 0.8535156265625 0.4055506828125 0.859375 0.4733307421875 0.828125 0.5038317671875 0.8662109375 0.59533484375 0.857421875 0.6653742375 0.8242187515625 0.7500993140625 0.7431640625 0.7952860171875 0.6337890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_359_jpg.rf.572476bc6f188b37294cd551c73bff36.txt
================================================
0 0.3606494953125 0.5439453125 0.362875728125 0.5263671859375 0.350631453125 0.515625 0.3439527625 0.537109375 0.307219940625 0.5380859375 0.3339347203125 0.5322265609375 0.3228035640625 0.5185546875 0.3417265296875 0.5019531265625 0.37178065312499997 0.5146484390625 0.3695544234375 0.4287109375 0.38513804218750003 0.4033203125 0.3740068859375 0.3798828140625 0.4485856390625 0.3085937515625 0.5532185171874999 0.3164062484375 0.607761190625 0.3486328140625 0.6099874187500001 0.4033203125 0.5543316359375 0.4384765609375 0.6177792281250001 0.478515625 0.7335432671875 0.4960937515625 0.8359499156249999 0.4707031265625 0.8615515765625 0.3544921859375 0.8103482546875 0.2646484390625 0.75135311875 0.21875 0.6734350171875 0.1855468734375 0.5977431484375 0.1914062484375 0.603308725 0.1591796875 0.5688021390625 0.134765625 0.5242775078124999 0.1328124984375 0.4975627328125 0.11328124843750001 0.45749056406250005 0.12109375156249999 0.3907036234375 0.09179687343750001 0.3573101484375 0.12109375156249999 0.3216904453125 0.11718750156249999 0.3061068265625 0.1367187515625 0.2259624921875 0.15625 0.1380263515625 0.2412109375 0.086823025 0.3173828140625 0.084596796875 0.3955078140625 0.12132961406250001 0.4492187515625 0.1814378640625 0.4707031265625 0.17253293593750002 0.484375 0.20036083124999998 0.4853515609375 0.2137182203125 0.5068359375 0.1981346015625 0.5146484390625 0.2415461140625 0.5332031265625 0.26158219531250004 0.5664062484375 0.2838445125 0.5507812484375 0.31167240625 0.5654296875 0.2860707421875 0.5761718734375 0.34283964375 0.5771484390625 0.31835109687499996 0.5654296875 0.348405225 0.5625 0.3383871828125 0.5498046875 0.3606494953125 0.5439453125


================================================
FILE: TumorDetection/valid/labels/no_tumor_362_jpg.rf.a97885865534be610f8dbbd58e56a432.txt
================================================
0 0.7324218734375 0.1826171859375 0.6396484359375 0.125 0.5478515640625 0.11328125 0.3505859359375 0.16015625 0.2802734359375 0.20703125 0.2167968734375 0.2783203140625 0.18359375 0.3505859359375 0.1875 0.4189453140625 0.22265625 0.4814453140625 0.2861328140625 0.51171875 0.3623046859375 0.51171875 0.4794921859375 0.4628906265625 0.5224609359375 0.546875 0.515625 0.4892578140625 0.5810546859375 0.4550781265625 0.6171875 0.5986328140625 0.65625 0.6533203140625 0.7636718734375 0.5673828140625 0.7949218734375 0.4892578140625 0.796875 0.3564453140625 0.7324218734375 0.1826171859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_367_jpg.rf.94625e3cea0d1413b37479452bfc4db0.txt
================================================
0 0.9413896984375001 0.8310546875 0.9227688046875 0.6572265640625 0.968286546875 0.4970703125 0.9641485671875 0.3056640640625 0.9331137453125 0.2138671875 0.8534576953125 0.11523437656249999 0.758284228125 0.048828123437499996 0.6569038015625 0.0234375 0.5658683140625 0.019531248437500003 0.416901153125 0.0507812484375 0.2906928640625 0.11132812343750001 0.19862288125 0.1845703125 0.1510361484375 0.2451171875 0.1241393015625 0.3173828125 0.11379436093749999 0.4150390640625 0.13241525468749998 0.5283203125 0.0434487546875 0.6513671875 0.099311440625 0.7255859359375 0.101380428125 0.8115234359375 0.17689850468749999 0.9746093765625 0.461384403125 0.9833984359375 0.919665321875 0.9785156234375 0.9413896984375001 0.9638671875 0.9269067796874999 0.9130859359375 0.9413896984375001 0.8310546875


================================================
FILE: TumorDetection/valid/labels/no_tumor_368_jpg.rf.eb89db210e555ab0c35246d0a8c794b7.txt
================================================
0 0.94921875 0.4794921859375 0.91015625 0.2490234359375 0.8613281265625 0.1572265640625 0.7470703140625 0.0605468734375 0.6044921859375 0.021484373437500003 0.4111328140625 0.05078125 0.2841796859375 0.109375 0.1347656265625 0.2626953140625 0.10546875 0.4423828140625 0.11328125 0.5244140640625 0.033203126562500004 0.6923828140625 0.047851564062500004 0.73828125 0.10742187343750001 0.7509765640625 0.1542968734375 0.9990234359375 0.859375 0.9990234359375 0.796875 0.9111328140625 0.796875 0.8173828140625 0.9199218734375 0.5869140640625 0.94921875 0.4794921859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_370_jpg.rf.a375d8d09f8ea0274c2972816b7289d0.txt
================================================
0 0.3642578140625 0.8401323890624999 0.4033203140625 0.796990453125 0.4130859359375 0.8355911328125 0.4140625 0.7754194875 0.4365234359375 0.7447660093749999 0.4501953140625 0.785637315625 0.4775390640625 0.769742915625 0.5224609359375 0.776554803125 0.5244140640625 0.756119153125 0.5517578140625 0.7606604046875 0.5576171859375 0.7447660093749999 0.5732421859375 0.762931034375 0.5947265640625 0.710706590625 0.59375 0.77996074375 0.61328125 0.7913138859375 0.6220703140625 0.8673799265625 0.6416015640625 0.8446736453125 0.6533203140625 0.883274325 0.6660156265625 0.8707858671875 0.6386718734375 0.8367264484374999 0.6484375 0.8276439359375001 0.625 0.8208320499999999 0.6152343734375 0.7867726312500001 0.6279296859375 0.7833666890625001 0.6689453140625 0.8401323890624999 0.8701171859375 0.8537561578125 0.9375 0.7572544625 0.9472656265625 0.6346405468749999 0.9296875 0.37578894531249996 0.8886718734375 0.2577162859375 0.7470703140625 0.118072659375 0.6513671859375 0.118072659375 0.5869140640625 0.0885544921875 0.5537109359375 0.093095753125 0.5185546859375 0.156673340625 0.4833984359375 0.084013240625 0.3447265640625 0.09990763593749999 0.1982421859375 0.18846212812500002 0.10351562656249999 0.307670103125 0.09375 0.3712476890625 0.1171875 0.3984952265625 0.0859375 0.41893088125 0.07226562656249999 0.568792334375 0.09765625 0.75952509375 0.16015625 0.877597753125 0.1953125 0.8980334046875 0.2060546859375 0.8469442765625 0.2275390640625 0.8719211828125 0.3017578140625 0.8673799265625 0.3642578140625 0.8401323890624999


================================================
FILE: TumorDetection/valid/labels/no_tumor_384_jpg.rf.add95791956e5b87c5a99cbdcebc52f3.txt
================================================
0 0.8710937484375 0.528365734375 0.8769531234375 0.44724291562499996 0.7675781234375 0.22522256718749997 0.6982421875 0.15797601875 0.5576171875 0.132358284375 0.4130859359375 0.1515715859375 0.2763671875 0.2241551578125 0.1953125015625 0.35117641875 0.2011718765625 0.4664562140625 0.3134765609375 0.5294331390625 0.5166015609375 0.5016805953125 0.5292968765625 0.5198264890625001 0.5078125015625 0.5582530890624999 0.5117187484375 0.6052189296875 0.5644531234375 0.71409429375 0.5791015609375 0.732240190625 0.6162109359375 0.7258357578125 0.6484374984375 0.7439816515625 0.7685546890625 0.6980832140625 0.7988281234375 0.6351062890625 0.8574218765625 0.5731967671875 0.8710937484375 0.528365734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_388_jpg.rf.6ae31956bef019bd6b92745808f5e29c.txt
================================================
0 0.8025970796875 0.7841796875 0.8358754968750001 0.6845703125 0.8554510343749999 0.3994140640625 0.8025970796875 0.2294921875 0.70765571875 0.12109375156249999 0.6587168734375 0.08984375156249999 0.5725845031250001 0.0585937515625 0.5236456546875 0.048828125 0.43555573281250004 0.048828125 0.37095645625 0.064453125 0.2926543015625 0.103515625 0.1663920765625 0.2177734359375 0.131156109375 0.3212890640625 0.12528344687499998 0.4755859359375 0.13898632500000002 0.6298828125 0.15268920156250002 0.7197265640625 0.16834963281249998 0.7568359359375 0.21728847812500002 0.8505859359375 0.255460778125 0.890625 0.30244207031250003 0.9257812484375 0.367041346875 0.9570312484375 0.47470680937500004 0.974609375 0.5324546484375 0.9736328125 0.59999025625 0.962890625 0.6704621953125 0.9335937515625 0.7360402500000001 0.8818359359375 0.8025970796875 0.7841796875


================================================
FILE: TumorDetection/valid/labels/no_tumor_392_jpg.rf.cd8a53dff0db8fe15f7b7198b14e04bd.txt
================================================
0 0.7890625 0.7687818484375 0.8535156265625 0.6525123640625 0.89453125 0.47409884687500004 0.8925781265625 0.4259873359375 0.87890625 0.3478061296875 0.8574218734375 0.301699265625 0.84765625 0.22953199999999999 0.7607421875 0.116269484375 0.6884765625 0.06013938906250001 0.5966796875 0.0220511078125 0.5166015625 0.008018584375 0.4326171875 0.010023232812500001 0.3173828125 0.040092925 0.2744140625 0.06013938906250001 0.2099609375 0.1002323140625 0.14453125 0.17139725625 0.10546875 0.28365744843749996 0.08984375 0.38589440937499997 0.08984375 0.4520477359375 0.13671875 0.6364751937500001 0.18359375 0.74472609375 0.2109375 0.7727911421875 0.2128906265625 0.8008561890625 0.23046875 0.8409491156250001 0.26171875 0.8830466875 0.3330078125 0.9441883984375 0.4033203125 0.9802720328125 0.55078125 0.989292940625 0.6142578125 0.9742580937499999 0.6728515625 0.9441883984375 0.73828125 0.8850513328125 0.7890625 0.7687818484375


================================================
FILE: TumorDetection/valid/labels/no_tumor_40_jpg.rf.08020a56bb66d390b1f9061b27f9ada3.txt
================================================
0 0.7822265640625 0.65625 0.8271484359375 0.623046875 0.8496093734375 0.5830078140625 0.84375 0.4697265640625 0.7597656265625 0.2958984359375 0.6943359375 0.2460937515625 0.6748046890625 0.2539062484375 0.6533203109375 0.2304687515625 0.6337890625 0.2402343734375 0.5556640625 0.2128906265625 0.4814453109375 0.2265624984375 0.4560546890625 0.251953125 0.3857421859375 0.2402343734375 0.2119140625 0.3339843734375 0.1679687515625 0.4072265640625 0.171875 0.4912109375 0.2724609375 0.5390624984375 0.2890624984375 0.6025390625 0.3623046890625 0.654296875 0.4033203109375 0.6640624984375 0.4560546890625 0.623046875 0.5263671859375 0.611328125 0.5117187515625 0.6318359375 0.515625 0.6923828140625 0.5654296890625 0.7539062484375 0.611328125 0.7646484359375 0.6884765640625 0.7402343734375 0.748046875 0.6884765640625 0.7646484359375 0.6484375015625 0.7822265640625 0.65625


================================================
FILE: TumorDetection/valid/labels/no_tumor_41_jpg.rf.fc79c11369566f35c789381021daacea.txt
================================================
0 0.8446438046875 0.5068359375 0.808444784375 0.3232421890625 0.7340356859375 0.1845703125 0.6385104953125 0.10546874843750001 0.49170335625000006 0.083984375 0.3509293890625 0.12304687656249999 0.245348915625 0.2314453125 0.1850172125 0.4013671890625 0.1709398171875 0.5615234359375 0.1990946109375 0.7021484359375 0.2815479328125 0.8330078109375 0.346907275 0.8886718765625 0.415283203125 0.9199218765625 0.5268968468749999 0.9326171890625 0.6606321187499999 0.896484375 0.7601794234375 0.7958984359375 0.83056640625 0.6494140625 0.8446438046875 0.5068359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_429_jpg.rf.ad8f82a83a75c8f3ea069bacb15b2667.txt
================================================
0 0.7470703140625 0.7998046875 0.8056640640625 0.6826171875 0.8307756703125 0.5751953125 0.8328683031249999 0.4775390625 0.7993861609375 0.3388671875 0.7407924109375 0.2138671875 0.6842912937500001 0.1435546875 0.6392996640625 0.10546875 0.5660574781250001 0.07421875 0.45724051406250005 0.0703125 0.3986467640625 0.08398437343750001 0.3337751109375 0.11328125 0.27622767812500004 0.1630859375 0.2280970984375 0.2353515625 0.16950334843749998 0.4130859375 0.1653180796875 0.6044921875 0.1715959828125 0.6474609375 0.2092633921875 0.7197265625 0.21763392812500001 0.7548828125 0.2898298 0.8535156265625 0.3274972109375 0.8828125 0.40910993281249997 0.9121093734375 0.5147879468750001 0.9150390625 0.5765206468749999 0.9121093734375 0.6288364953125 0.8964843734375 0.6874302453125 0.85546875 0.7470703140625 0.7998046875


================================================
FILE: TumorDetection/valid/labels/no_tumor_436_jpg.rf.94fb51400d8e9579600f4640fa4a1842.txt
================================================
0 0.8468511453125 0.7861328125 0.8826335875 0.6083984375 0.8866094140625 0.5263671875 0.8627544531250001 0.3720703125 0.838899490625 0.2939453125 0.79516539375 0.2001953125 0.7514312984375 0.1357421875 0.7027274171875 0.08984375 0.6212229640625 0.03515625 0.5834526078125 0.025390625 0.47411736718749997 0.0234375 0.422431615625 0.03515625 0.3369513359375 0.08984375 0.29023536875 0.1357421875 0.246501271875 0.2099609375 0.22662213750000001 0.2744140625 0.16300890625 0.4111328125 0.13915394375 0.5537109375 0.1530693390625 0.7646484375 0.194815521875 0.8330078125 0.26737436406250004 0.904296875 0.3309875953125 0.94921875 0.410504134375 0.98046875 0.508905853125 0.9873046875 0.5675493 0.982421875 0.6530295796875001 0.95703125 0.7514312984375 0.8935546875 0.8468511453125 0.7861328125


================================================
FILE: TumorDetection/valid/labels/no_tumor_456_jpg.rf.90e088657683fff3d66e7f5375a7599d.txt
================================================
0 0.8333705374999999 0.6103515625 0.8465290171875001 0.4873046875 0.8289843749999999 0.4345703125 0.8070535718749999 0.4189453125 0.7653850453125 0.3525390640625 0.7368750000000001 0.3349609359375 0.7500334828125 0.3173828109375 0.7500334828125 0.2666015625 0.7281026796875001 0.2177734375 0.6765652921875 0.173828125 0.6261244421875001 0.171875 0.5888420765625 0.1347656265625 0.5581389515625 0.126953125 0.5252427453124999 0.1347656265625 0.44629185156249995 0.12890625 0.37392020000000004 0.15234375 0.3508928578125 0.1923828109375 0.303741628125 0.21484375 0.27852120625 0.2431640640625 0.2741350453125 0.2783203125 0.30922433125 0.3095703125 0.24781807968749997 0.3310546875 0.17763950781249999 0.4560546875 0.164481025 0.5087890640625 0.1666741078125 0.5830078109375 0.2302734375 0.7275390640625 0.29277622812500004 0.78125 0.39585100624999997 0.830078125 0.426554128125 0.828125 0.45067801406249997 0.798828125 0.4769949765625 0.796875 0.4791880578125 0.783203125 0.4824776796875 0.8095703125 0.5011188625 0.828125 0.5482700890625 0.8369140640625 0.5888420765625 0.83203125 0.672179128125 0.79296875 0.778543525 0.7060546875 0.8333705374999999 0.6103515625


================================================
FILE: TumorDetection/valid/labels/no_tumor_50_jpg.rf.3be65f9c232a4652ded849dbc661cddd.txt
================================================
0 0.932762634375 0.7626953140625 0.9631316515625 0.5732421875 0.9414394937499999 0.3896484375 0.837317153125 0.1845703140625 0.7820021609375 0.12109375156249999 0.68655668125 0.0585937515625 0.54555768125 0.03125 0.3394822125 0.0585937515625 0.2548828125 0.099609375 0.16702958750000002 0.1845703140625 0.0802609703125 0.3388671875 0.04338430625 0.4638671875 0.04121509375 0.6201171875 0.060738034375000007 0.7314453140625 0.1431682171875 0.8818359390625 0.2375290890625 0.9609375015625 0.3546667234375 1 0.642087765625 0.9990234375 0.7386178515624999 0.9648437515625 0.7993558843749999 0.9238281265625 0.8850398937499999 0.8427734375 0.932762634375 0.7626953140625


================================================
FILE: TumorDetection/valid/labels/no_tumor_535_jpg.rf.a70887aedbcf0828ad42060326e941f2.txt
================================================
0 0.755859375 0.4094358359375 0.724609375 0.250944546875 0.6259765640625 0.09098574375 0.5380859375 0.041090332812500005 0.4287109375 0.0469603828125 0.3681640625 0.0821806671875 0.275390625 0.245074496875 0.2460937484375 0.4622662671875 0.244140625 0.682393059375 0.287109375 0.805664059375 0.3564453125 0.915727459375 0.4150390625 0.9568177921875 0.5449218765625 0.9670903765625001 0.6337890625 0.9274675515625 0.728515625 0.7968589890625 0.7636718765625 0.635432678125 0.755859375 0.4094358359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_546_jpg.rf.cd7b37e7b4749b28f6d52c8ad6889292.txt
================================================
0 0.8183593734375 0.44254328125000003 0.7890625 0.3256858875 0.70703125 0.17853213125 0.5966796859375 0.064920775 0.5224609359375 0.0302963609375 0.4462890640625 0.0216402578125 0.3310546859375 0.06708479843750001 0.16796875 0.2650931640625 0.12109375 0.4187389953125 0.11914062656249999 0.5680567796875 0.14453125 0.687078196875 0.2177734359375 0.8418060468749999 0.3564453140625 0.9738116203124999 0.53125 0.9900418125 0.6826171859375 0.9002347421874999 0.7519531265625 0.7996075406250001 0.8066406265625 0.669765990625 0.8183593734375 0.44254328125000003


================================================
FILE: TumorDetection/valid/labels/no_tumor_580_jpg.rf.a6973334fe566d120d5e89e43012ee12.txt
================================================
0 0.9469259515624999 0.6982421890625 0.9675611406250001 0.5576171890625 0.9583899468749999 0.3994140625 0.8506283968749999 0.1826171890625 0.7508916421875 0.09375 0.6156165078125 0.042968748437499996 0.429899796875 0.052734375 0.30379585468750003 0.11718749843750001 0.2040591015625 0.2138671890625 0.139860734375 0.3232421890625 0.10317595156249999 0.4482421890625 0.1467391296875 0.7236328109375 0.2338654875 0.8486328109375 0.4230214015625 0.947265625 0.5548573375 0.9619140625 0.6247877046875 0.958984375 0.72796365625 0.9257812515625 0.8483355984375001 0.8486328109375 0.9469259515624999 0.6982421890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_613_jpg.rf.03e2f7f23767650797806734a2466921.txt
================================================
0 0.9011829296875 0.6669921890625 0.909953565625 0.6025390625 0.88364165625 0.3544921890625 0.8485591078124999 0.2529296890625 0.7356371609375 0.1269531265625 0.595306971875 0.08789062656249999 0.3870043484375 0.0859375 0.24228883906249998 0.1503906265625 0.1512934859375 0.2548828109375 0.11840359375000001 0.3408203109375 0.083321046875 0.6201171890625 0.1885686890625 0.7626953109375 0.2861420234375 0.8398437484375 0.4155089203125 0.8847656265625 0.54377948125 0.8876953109375 0.6457381328125 0.8730468734375 0.7224812078125 0.8378906265625 0.8134765609375 0.7666015625 0.9011829296875 0.6669921890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_615_jpg.rf.69a773dbaf49fcd6b71f890fdb8fc5e7.txt
================================================
0 0.765625 0.8059807859375001 0.8144531234375 0.6742240296875 0.8613281234375 0.48482369218750004 0.8671875 0.4230627109375 0.8613281234375 0.32218644375 0.8222656234375 0.178077490625 0.7988281234375 0.126610009375 0.7587890625 0.07617187343750001 0.6787109375 0.0123521953125 0.6611328125 0.0041173984375 0.5908203125 0 0.3681640625 0.0041173984375 0.3212890625 0.026763090625000003 0.28515625 0.060731631249999994 0.2285156234375 0.143079603125 0.19921875 0.2027818828125 0.16796875 0.3324799421875 0.16796875 0.43747360625000004 0.17578125 0.47864759218749997 0.23828125 0.6906936234375001 0.2792968765625 0.7956872890625 0.30859375 0.845096071875 0.3505859375 0.8852407093749999 0.4189453125 0.9243559953125 0.4755859375 0.9387668921874999 0.55859375 0.9418549421875 0.6064453125 0.93464949375 0.6748046875 0.9017103046875 0.7099609375 0.8728885140625 0.765625 0.8059807859375001


================================================
FILE: TumorDetection/valid/labels/no_tumor_622_jpg.rf.4ec6d2dee2d9178e74a0031e87d9d48b.txt
================================================
0 0.8535156265625 0.4287109375 0.8007812484375 0.2763671859375 0.7060546890625 0.1523437515625 0.5830078140625 0.08984375156249999 0.4189453109375 0.08398437343750001 0.2783203109375 0.1523437515625 0.1914062484375 0.2568359375 0.1523437515625 0.3525390625 0.1445312484375 0.5400390625 0.1816406265625 0.7412109375 0.2412109375 0.8242187515625 0.4013671859375 0.9101562484375 0.53125 0.9169921859375 0.6533203109375 0.892578125 0.779296875 0.7939453109375 0.8398437515625 0.6689453109375 0.8535156265625 0.4287109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_62_jpg.rf.ef72cd835b012e3ecc8829d4a0fae30d.txt
================================================
0 0.945461465625 0.6044921890625 0.985020525 0.4716796859375 0.9771087125 0.3720703140625 0.9039244578125001 0.2158203140625 0.7644787796875 0.07031250156249999 0.641845703125 0.023437498437500003 0.378777975 0.0585937484375 0.20274016874999998 0.1367187484375 0.0731842546875 0.2724609375 0.0415370078125 0.3642578109375 0.053404728125 0.5068359375 0.11867716875 0.5947265609375 0.100875596875 0.6728515609375 0.10680945312500001 0.7314453140625 0.0731842546875 0.7548828109375 0.0731842546875 0.7724609375 0.1028535515625 0.7939453140625 0.0870299234375 0.8642578109375 0.134500796875 0.8876953140625 0.13647874531249998 0.9423828109375 0.18296063906249999 1 0.840629965625 0.9990234390625 0.7990929578125 0.9248046859375 0.785247284375 0.8115234390625 0.8070047671875 0.7587890625 0.945461465625 0.6044921890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_639_jpg.rf.9a390098e900f70cc25290f5c2b2acfc.txt
================================================
0 0.8535156265625 0.5791015640625 0.841796875 0.3447265640625 0.826171875 0.2783203109375 0.7773437515625 0.1962890625 0.6689453109375 0.11328124843750001 0.5419921859375 0.08203124843750001 0.4072265640625 0.08593750156249999 0.2548828140625 0.1640624984375 0.1816406265625 0.2587890625 0.142578125 0.4501953109375 0.1523437515625 0.6552734359375 0.21875 0.7880859375 0.3095703109375 0.8789062484375 0.4130859375 0.919921875 0.5351562484375 0.9228515640625 0.6806640625 0.8789062484375 0.7382812484375 0.8291015640625 0.8164062484375 0.7060546890625 0.8535156265625 0.5791015640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_642_jpg.rf.512324112aae0a5ec37a1d955b6ec2d9.txt
================================================
0 0.9508115921875 0.5654296890625 0.9693188734374999 0.3818359390625 0.8582751843749999 0.1982421890625 0.7391345578124999 0.10546875 0.5702556109375 0.0527343765625 0.39674985 0.048828123437499996 0.2579452375 0.09570312343750001 0.13186437812500001 0.2021484390625 0.0416413828125 0.4013671890625 0.0347011546875 0.5361328109375 0.1295509703125 0.7802734390625 0.2255574953125 0.8984375 0.3921230296875 0.9628906234375 0.5598452671875 0.9716796890625 0.6651054328125 0.9511718765625 0.7576418390625 0.9042968765625 0.8883495140625 0.7626953109375 0.9508115921875 0.5654296890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_643_jpg.rf.73be8e13c31c16c3f3de6d73d81f5a41.txt
================================================
0 0.9446331499999999 0.6923828109375 0.9675611406250001 0.4521484359375 0.9194123624999999 0.3095703125 0.8483355984375001 0.1845703125 0.75318444375 0.09765625156249999 0.647715690625 0.048828123437499996 0.45741338125000003 0.048828123437499996 0.3267238453125 0.103515625 0.21552309687500001 0.1982421890625 0.12381114062500001 0.3720703125 0.10546874843750001 0.4931640625 0.14215353125000002 0.7060546875 0.23157269062500002 0.8427734359375 0.41155740625000004 0.9414062515625 0.56402853125 0.9599609375 0.7256708546875 0.9257812515625 0.80821161875 0.880859375 0.8689707875 0.8212890625 0.9446331499999999 0.6923828109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_64_jpg.rf.c37f888a9cd7193a8282b28ad1e821b1.txt
================================================
0 0.8085937515625 0.13543991875 0.7275390640625 0.0650111625 0.5810546859375 0.047674853125 0.3857421890625 0.12352120468750001 0.2451171890625 0.22970609999999997 0.1757812484375 0.3608119421875 0.140625 0.5211728046874999 0.171875 0.6490280859374999 0.09375 0.8267252624999999 0.10644531406249999 0.8689825140625 0.15625 0.8787341906249999 0.1621093734375 0.9935872375000001 0.904296875 0.9957542765625 0.951171875 0.59485211875 0.9414062484375 0.4691638765625 0.888671875 0.2676292765625 0.8085937515625 0.13543991875


================================================
FILE: TumorDetection/valid/labels/no_tumor_65_jpg.rf.d0c617db44ca384b082d71b7bf4407c0.txt
================================================
0 0.9731518203125 0.7275390640625 0.99022465625 0.5556640640625 0.924372278125 0.3232421875 0.841447059375 0.1845703109375 0.6938889546875 0.07421874843750001 0.579257034375 0.035156251562500004 0.328042403125 0.041015625 0.1853622484375 0.10644531093750001 0.0536574921875 0.2880859359375 0.009755910937499999 0.4931640640625 0.0317067015625 0.7021484375 0.10975396562499999 0.8466796890625 0.17682583125 0.9140625015625 0.289018771875 0.9726562515625 0.3816999 1 0.6755966265625 0.9990234375 0.77681416875 0.96875 0.860958875 0.9150390640625 0.9316892093750001 0.8212890640625 0.9731518203125 0.7275390640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_665_jpg.rf.be649898c2cdde1292dc507e28da0927.txt
================================================
0 0.8535156265625 0.648503071875 0.8925781265625 0.47409884687500004 0.8886718734375 0.4059408734375 0.8574218734375 0.3097178515625 0.84765625 0.23354129218750003 0.7646484375 0.122283425 0.6884765625 0.06414868124999999 0.6103515625 0.030069693749999998 0.5458984375 0.012027878124999999 0.4326171875 0.012027878124999999 0.3193359375 0.04410221875 0.2294921875 0.0882044359375 0.1503906265625 0.1673879640625 0.11328125 0.2555924 0.09375 0.3818851171875 0.09570312656249999 0.45806167656249996 0.13671875 0.6324659015625 0.1855468734375 0.746730740625 0.2089843734375 0.7687818484375 0.21484375 0.80486548125 0.25390625 0.8710188109375 0.3349609375 0.9441883984375 0.3994140625 0.9782673859375001 0.5449218734375 0.989292940625 0.6435546875 0.96022556875 0.6943359375 0.9281512281250001 0.73828125 0.8830466875 0.8535156265625 0.648503071875


================================================
FILE: TumorDetection/valid/labels/no_tumor_675_jpg.rf.1ad1f8846e1d8537cb1f0d83afce820d.txt
================================================
0 0.9328358203124999 0.5322265640625 0.9352650796875001 0.4033203125 0.9036847015625 0.2744140640625 0.8247337515625001 0.1523437515625 0.732421875 0.09179687656249999 0.550227378125 0.048828123437499996 0.479778840625 0.048828123437499996 0.2830087859375 0.1015625 0.18948227812500001 0.1689453125 0.1068874375 0.2900390640625 0.07044853749999999 0.4794921875 0.1651896765625 0.7216796875 0.23320895468749997 0.8330078125 0.3631743625 0.9121093765625 0.5684468265625 0.9248046875 0.749426696875 0.8613281234375 0.823519125 0.7724609359375 0.855099503125 0.6533203125 0.9328358203124999 0.5322265640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_677_jpg.rf.c3e61ad4137f047ad843e240c03858c8.txt
================================================
0 0.767578125 0.5908203109375 0.8066406265625 0.5166015640625 0.8125 0.4326171859375 0.798828125 0.3583984359375 0.7578124984375 0.2763671859375 0.6708984359375 0.1875 0.5830078140625 0.1484375015625 0.4462890625 0.138671875 0.3759765640625 0.158203125 0.283203125 0.2255859375 0.201171875 0.3681640625 0.1875 0.4775390625 0.2402343734375 0.5986328140625 0.25 0.7177734359375 0.2851562484375 0.7724609375 0.3623046890625 0.8359375015625 0.4169921859375 0.8554687515625 0.513671875 0.8583984359375 0.6376953109375 0.8378906265625 0.71875 0.7705078140625 0.751953125 0.7119140625 0.767578125 0.5908203109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_683_jpg.rf.c44df82d787ccf30e1d50529d8beb588.txt
================================================
0 0.6044921875 0.8046875 0.6083984375 0.794921875 0.6533203125 0.783203125 0.6669921875 0.775390625 0.685546875 0.7548828125 0.69140625 0.7392578125 0.7109375 0.7255859375 0.7265625 0.6865234375 0.73046875 0.6396484375 0.724609375 0.6259765625 0.751953125 0.5576171875 0.75390625 0.5283203125 0.771484375 0.4892578125 0.771484375 0.4638671875 0.78515625 0.4130859375 0.78515625 0.3818359375 0.779296875 0.3642578125 0.78515625 0.3427734375 0.783203125 0.3251953125 0.767578125 0.2744140625 0.73828125 0.2255859375 0.6708984375 0.154296875 0.5908203125 0.12109375 0.5615234375 0.123046875 0.5283203125 0.1171875 0.5107421875 0.123046875 0.4990234375 0.134765625 0.4775390625 0.119140625 0.4638671875 0.1171875 0.4462890625 0.123046875 0.4091796875 0.15234375 0.3837890625 0.150390625 0.3720703125 0.15625 0.3330078125 0.193359375 0.3095703125 0.203125 0.2890625 0.2255859375 0.263671875 0.2705078125 0.240234375 0.3564453125 0.240234375 0.4423828125 0.255859375 0.4736328125 0.251953125 0.4833984375 0.255859375 0.5126953125 0.271484375 0.5419921875 0.275390625 0.5712890625 0.2890625 0.5966796875 0.291015625 0.6708984375 0.314453125 0.7177734375 0.31640625 0.7333984375 0.361328125 0.7646484375 0.3681640625 0.7890625 0.4521484375 0.81640625 0.4736328125 0.830078125 0.501953125 0.8291015625 0.5126953125 0.82421875 0.5439453125 0.826171875 0.5751953125 0.80859375 0.6044921875 0.8046875


================================================
FILE: TumorDetection/valid/labels/no_tumor_741_jpg.rf.8295ca43d5df8144af439d0869af0881.txt
================================================
0 0.8886718765625 0.5093868640625 0.8789062515625 0.40643709843750003 0.8476562515625 0.3056321171875 0.7294921875 0.1136737015625 0.6630859359375 0.06434360312500001 0.5888671875 0.0386061640625 0.4013671875 0.0386061640625 0.2646484390625 0.08793625625 0.1367187484375 0.25201244843750004 0.08984374843750001 0.36997572500000003 0.0605468765625 0.5093868640625 0.06640625156249999 0.7281551171875 0.11328125156249999 0.8611319015625 0.2783203109375 0.9587197015625 0.4492187484375 0.9726608125 0.6767578125 0.9501405515625001 0.8271484390625 0.8493355703125 0.8789062515625 0.7474582015625 0.8886718765625 0.5093868640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_771_jpg.rf.e40daaf725f1f6817dfeeeff5c66a145.txt
================================================
0 0.857421875 0.5029296890625 0.8496093734375 0.3369140625 0.814453125 0.2626953109375 0.7001953109375 0.1640624984375 0.5478515640625 0.11718750156249999 0.4345703109375 0.11718750156249999 0.3232421859375 0.1484375015625 0.2197265640625 0.220703125 0.1503906265625 0.2978515640625 0.140625 0.3740234359375 0.1484375015625 0.6005859375 0.2597656265625 0.8642578140625 0.2763671859375 0.8828124984375 0.2861328140625 0.8671875015625 0.2919921859375 0.888671875 0.3173828140625 0.8828124984375 0.3369140625 0.9140624984375 0.3505859375 0.9003906265625 0.3857421859375 0.921875 0.4746093734375 0.9228515640625 0.6318359375 0.9160156265625 0.6953124984375 0.8798828140625 0.8046875015625 0.7021484359375 0.857421875 0.5029296890625


================================================
FILE: TumorDetection/valid/labels/no_tumor_777_jpg.rf.6d02a1ac2cd70272de1cc347ff43fa0e.txt
================================================
0 0.9511523453124999 0.734375 0.9437695296875 0.7109375 0.9179296906250001 0.7099609359375 0.9302343750000001 0.6142578140625 0.945 0.6083984359375 0.945 0.5556640640625 0.9302343750000001 0.5517578140625 0.95484375 0.5263671859375 0.9388476546875 0.4765625 0.927773440625 0.5283203140625 0.9080859406249999 0.4384765640625 0.9560742203125001 0.4199218734375 0.9253125000000001 0.4072265640625 0.9573046906249999 0.3994140640625 0.905625 0.3935546859375 0.959765625 0.3857421859375 0.9093164046875 0.3808593734375 0.9203906249999999 0.3720703140625 0.8994726546875 0.3496093734375 0.8859375 0.3623046859375 0.8637890593750001 0.2666015640625 0.821953125 0.2021484359375 0.826875 0.1728515640625 0.8071875000000001 0.1865234359375 0.7924218749999999 0.1611328140625 0.8047265593749999 0.1513671859375 0.7911914046875 0.1582031265625 0.735820309375 0.11425781406249999 0.7468945296875 0.109375 0.7272070296875001 0.109375 0.725976559375 0.08886718593750001 0.7099804703125 0.09179687343750001 0.681679690625 0.0556640640625 0.7641210953125 0.08203125 0.7468945296875 0.0605468734375 0.5992382796875 0.0078125 0.4614257796875 0.0058593734375 0.3654492203125 0.0234375 0.20671875 0.10449218593750001 0.06890625 0.2919921859375 0.0196875 0.4814453140625 0.017226559375 0.5166015640625 0.036914059375 0.5224609359375 0.014765625 0.5263671859375 0.00984375 0.5751953140625 0.039375 0.7705078140625 0.0725976546875 0.8183593734375 0.103359375 0.8193359359375 0.051679690625 0.7333984359375 0.0725976546875 0.71875 0.10828125 0.7861328140625 0.209179690625 0.8759765640625 0.17349609531250001 0.890625 0.1858007796875 0.8769531265625 0.1538085953125 0.875 0.09966797031249999 0.82421875 0.08859375 0.8330078140625 0.2104101546875 0.9296875 0.260859375 0.9267578140625 0.228867190625 0.9130859359375 0.23748047031250002 0.8984375 0.2424023453125 0.91796875 0.2522460953125 0.9003906265625 0.29162109531249997 0.9082031265625 0.3285351546875 0.9472656265625 0.4269726546875 0.9472656265625 0.44666015468750003 0.9316406265625 0.4983398453125 0.9296875 0.5795507796875 0.94921875 0.6656835953125 0.94921875 0.7247460953125 0.9394531265625 0.7284375 0.9052734359375 0.7715039046875 0.9003906265625 0.753046875 0.9560546859375 0.8711718749999999 0.8798828140625 0.972070309375 0.7431640640625 0.9806835953125 0.66796875 0.9511523453124999 0.734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_778_jpg.rf.85b0a33e447c7fb1160ae25e6f255cbe.txt
================================================
0 0.9251093031250001 0.6552734375 0.916830921875 0.4169921859375 0.860951834375 0.2783203109375 0.744019675 0.1523437484375 0.6115655453125 0.10156250156249999 0.4190931421875 0.10351562343750001 0.278360628125 0.169921875 0.157289275 0.3076171859375 0.11175817031249999 0.4169921859375 0.0869230234375 0.5439453109375 0.095201403125 0.7197265625 0.18626361875 0.8857421859375 0.315613353125 0.9648437484375 0.4925637890625 0.9853515625 0.636400696875 0.970703125 0.779202803125 0.9121093765625 0.8568126421875 0.8271484375 0.9251093031250001 0.6552734375


================================================
FILE: TumorDetection/valid/labels/no_tumor_782_jpg.rf.82c6b020ac0f81fe14e28a880febfec3.txt
================================================
0 0.8339843734375 0.4326171859375 0.779296875 0.2783203109375 0.7021484359375 0.1835937515625 0.5712890625 0.11914062656249999 0.4326171859375 0.12109375156249999 0.2978515640625 0.1953124984375 0.21875 0.3232421859375 0.1835937515625 0.4638671859375 0.1914062484375 0.7431640625 0.2285156265625 0.8330078140625 0.3525390625 0.9453124984375 0.4365234359375 0.9765624984375 0.5664062484375 0.9814453109375 0.7041015640625 0.919921875 0.8085937515625 0.7939453109375 0.8320312484375 0.7275390625 0.8339843734375 0.4326171859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_789_jpg.rf.7bbf779a54237b93b2a590d5de20ee59.txt
================================================
0 0.9075917125 0.5029296875 0.8884341046875001 0.3857421875 0.75433084375 0.1982421875 0.64297724375 0.1328125 0.504084578125 0.11328124843750001 0.3412449046875 0.1386718765625 0.2203125 0.2138671875 0.1293138609375 0.3701171875 0.09339334375 0.5458984359375 0.1341032609375 0.7099609359375 0.167629078125 0.7705078125 0.2286939515625 0.8320312484375 0.3604025125 0.8945312484375 0.5579653515625 0.9072265640625 0.755528190625 0.8574218765625 0.8333559781250001 0.7919921875 0.8908288046875 0.6787109359375 0.9075917125 0.5029296875


================================================
FILE: TumorDetection/valid/labels/no_tumor_798_jpg.rf.d1d5b7d9bd8c9372eae7268d7d74f60a.txt
================================================
0 0.962378640625 0.5927734390625 0.9600652328124999 0.4189453109375 0.8883495140625 0.2275390609375 0.7622686609375 0.08984375 0.567942203125 0.0292968765625 0.4013766703125 0.03125 0.23712454374999997 0.09570312343750001 0.1642521265625 0.1591796890625 0.10179004843749999 0.2451171890625 0.046268203125000004 0.4306640609375 0.0416413828125 0.6591796890625 0.1503716609375 0.8115234390625 0.255631825 0.8925781234375 0.38980961718749996 0.9355468765625 0.592233009375 0.9462890609375 0.655851790625 0.9433593765625 0.78308935 0.8828125 0.867528825 0.8017578109375 0.962378640625 0.5927734390625


================================================
FILE: TumorDetection/valid/labels/no_tumor_803_jpg.rf.7e7b6949454153f02fece110715ba2bf.txt
================================================
0 0.903433865625 0.6357421875 0.903433865625 0.4794921875 0.839662065625 0.2802734359375 0.7386900421875 0.1523437515625 0.6005178046874999 0.08789062343750001 0.4134538515625 0.0859375 0.2816587953125 0.1601562484375 0.1955668609375 0.2802734359375 0.148800875 0.4228515640625 0.1509266 0.6318359359375 0.22745276250000002 0.7841796875 0.3177961484375 0.8710937515625 0.4474654796875 0.9179687515625 0.5930777625 0.9189453125 0.6749182421875 0.9042968765625 0.7429414984375 0.8710937515625 0.846039246875 0.7744140640625 0.903433865625 0.6357421875


================================================
FILE: TumorDetection/valid/labels/no_tumor_820_jpg.rf.09b1e08174791ce390a1d64357d82f1d.txt
================================================
0 0.94388409375 0.3232421875 0.8341301296875001 0.1416015625 0.662182253125 0.048828123437499996 0.55730624375 0.035156251562500004 0.3377983140625 0.042968748437499996 0.2109715078125 0.07617187656249999 0.09024215 0.1845703109375 0.04146260625 0.3544921875 0.0512185171875 0.6923828125 0.1658504375 0.8876953109375 0.2499951390625 0.9492187484375 0.29633570625 0.9394531234375 0.40365069218749994 0.974609375 0.5073072140625 0.9814453109375 0.7060838406250001 0.9335937484375 0.8438860390625 0.8115234375 0.9414451187499999 0.5830078125 0.94388409375 0.3232421875


================================================
FILE: TumorDetection/valid/labels/no_tumor_844_jpg.rf.5c4388e5461f6ba4346218e13e6ed9a8.txt
================================================
0 0.7734375 0.8251953125 0.8007812515625 0.7568359375 0.826171875 0.6630859375 0.8359375 0.5888671875 0.828125 0.4892578125 0.7890625 0.3017578125 0.7539062515625 0.2119140625 0.7304687484375 0.1787109375 0.6650390625 0.119140625 0.5966796875 0.083984375 0.5439453125 0.07421874843750001 0.4150390625 0.07421874843750001 0.3369140625 0.10546874843750001 0.2822265625 0.1484375 0.2460937484375 0.1943359375 0.2148437484375 0.2685546875 0.1875 0.3623046875 0.1679687484375 0.4892578125 0.1679687484375 0.6181640625 0.185546875 0.7119140625 0.2109375 0.7919921875 0.240234375 0.8388671875 0.2802734375 0.8828125 0.4033203125 0.947265625 0.5390625 0.9580078125 0.6044921875 0.9492187484375 0.6962890625 0.908203125 0.732421875 0.8818359375 0.7734375 0.8251953125


================================================
FILE: TumorDetection/valid/labels/no_tumor_84_jpg.rf.e0c80858940cf7d4e12b89756f9f71b3.txt
================================================
0 0.8211436171875001 0.7060546875 0.87925531875 0.5478515625 0.8615691484375001 0.4501953125 0.7781914890625 0.3173828125 0.7390292546875 0.291015625 0.6986037234375 0.283203125 0.665757978125 0.3125 0.6215425531250001 0.3251953125 0.5988031921875 0.3798828125 0.517952128125 0.4365234375 0.53816489375 0.4404296875 0.5419547875 0.46484375 0.5015292546875 0.470703125 0.5078457453125 0.4560546875 0.483843084375 0.453125 0.47878989375 0.466796875 0.44594414843749997 0.4609375 0.430784575 0.4765625 0.43710106406250004 0.4091796875 0.399202128125 0.3466796875 0.332247340625 0.28515625 0.2741356390625 0.283203125 0.1768617015625 0.3798828125 0.103590425 0.5693359375 0.1111702125 0.6533203125 0.15412234062500002 0.7099609375 0.2627659578125 0.7900390625 0.28297872343749997 0.8349609375 0.36003989375 0.890625 0.47626329843750004 0.927734375 0.581117021875 0.9287109375 0.6834441484375 0.890625 0.7554521281250001 0.8232421875 0.780718084375 0.7490234375 0.8211436171875001 0.7060546875


================================================
FILE: TumorDetection/valid/labels/no_tumor_854_jpg.rf.3753a0769e4956bf721130cd6f9caa83.txt
================================================
0 0.8476562484375 0.3994140625 0.8242187515625 0.2880859375 0.783203125 0.2138671859375 0.6748046890625 0.12109375156249999 0.5791015640625 0.08789062656249999 0.4091796890625 0.08789062656249999 0.2548828140625 0.1679687515625 0.1835937515625 0.2587890625 0.142578125 0.4599609375 0.1503906265625 0.6455078140625 0.1972656265625 0.7587890625 0.2939453109375 0.8652343734375 0.4111328140625 0.9179687515625 0.533203125 0.9208984359375 0.6591796890625 0.8847656265625 0.7753906265625 0.7705078140625 0.8476562484375 0.6103515640625 0.8476562484375 0.3994140625


================================================
FILE: TumorDetection/valid/labels/no_tumor_855_jpg.rf.a52d1a3997edfe6d0d665a98b8a88694.txt
================================================
0 0.70114978125 0.8125 0.7683558390624999 0.7470703125 0.8024604078124999 0.6865234390625 0.8104850109375 0.6474609390625 0.82653421875 0.6201171875 0.8185096156249999 0.4462890609375 0.7864111984375 0.3369140609375 0.7783865953125 0.2607421875 0.7503004796875 0.2158203125 0.742275878125 0.1806640609375 0.69011595 0.1123046875 0.6650390640625 0.09375 0.590811475 0.07421874843750001 0.55269460625 0.0546875 0.5165838875000001 0.064453125 0.450380903125 0.056640625 0.31095340781249997 0.1259765609375 0.2607996328125 0.2353515609375 0.2347196703125 0.3681640609375 0.21666430937499997 0.4033203125 0.21666430937499997 0.4345703125 0.19459664999999998 0.4794921875 0.1885781953125 0.5185546875 0.1865720453125 0.5615234390625 0.2006151015625 0.6357421875 0.23271351875000001 0.7119140609375 0.2628057828125 0.7568359390625 0.3199810875 0.8046875 0.38417792031250003 0.84375 0.43834399843750005 0.8671875 0.48147624375 0.8681640609375 0.5366453984375 0.8476562515625 0.6329406468750001 0.8398437484375 0.659020609375 0.822265625 0.70114978125 0.8125


================================================
FILE: TumorDetection/valid/labels/no_tumor_890_jpg.rf.eb4da32fb507e2296f266fc017910798.txt
================================================
0 0.841796875 0.3701171859375 0.7851562484375 0.1826171859375 0.7255859375 0.10156249843750001 0.6298828140625 0.048828125 0.4189453109375 0.035156248437499996 0.3056640625 0.07031249843750001 0.2255859375 0.123046875 0.1816406265625 0.1767578140625 0.15625 0.2412109375 0.1523437515625 0.5576171859375 0.2421875015625 0.7412109375 0.3310546890625 0.8515624984375 0.3779296890625 0.8828124984375 0.564453125 0.8916015640625 0.6650390625 0.8476562484375 0.8066406265625 0.6669921859375 0.845703125 0.4462890625 0.841796875 0.3701171859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_892_jpg.rf.895906826facdd5d906e9f997092d56d.txt
================================================
0 0.79672441875 0.7880859359375 0.8319603875 0.6865234359375 0.8515359281249999 0.5107421875 0.8515359281249999 0.3955078125 0.814342403125 0.2646484359375 0.7888942046875 0.2099609359375 0.705698165625 0.12109375156249999 0.6587168734375 0.08984375156249999 0.5627967328125 0.056640625 0.5040701171874999 0.046875 0.43359817968750003 0.048828125 0.37291400937499997 0.064453125 0.2946118546875 0.103515625 0.1644345234375 0.2255859359375 0.135071215625 0.3076171875 0.127241 0.4345703125 0.144858984375 0.6689453125 0.1546467546875 0.7197265640625 0.19967049375 0.8173828125 0.227076246875 0.8623046875 0.265248546875 0.8984375 0.367041346875 0.9570312484375 0.47470680937500004 0.974609375 0.5246244328125 0.9736328125 0.6195657953125 0.9570312484375 0.6919952875 0.9179687515625 0.728210034375 0.8876953125 0.79672441875 0.7880859359375


================================================
FILE: TumorDetection/valid/labels/no_tumor_905_jpg.rf.1c4afb0f72368adba5f7fdb315c90e62.txt
================================================
0 0.9643554703125 0.7470703125 0.9912109359375 0.5712890609375 0.9228515640625 0.3212890609375 0.84228515625 0.1865234375 0.6970214828125 0.076171875 0.5798339828125 0.035156251562500004 0.3234863265625 0.042968748437499996 0.1684570296875 0.1220703125 0.0512695296875 0.2939453125 0.009765626562500001 0.4951171859375 0.03662109375 0.7080078140625 0.10986327968750001 0.8447265625 0.1721191390625 0.90625 0.2624511734375 0.9589843765625 0.3967285171875 1 0.6640625 0.9990234375 0.7385253890625 0.984375 0.85693359375 0.9189453125 0.91796875 0.8408203125 0.9643554703125 0.7470703125


================================================
FILE: TumorDetection/valid/labels/no_tumor_907_jpg.rf.74a36007d3ef39b6458b7ddee7582e1b.txt
================================================
0 0.8361166875 0.4638671859375 0.790585578125 0.4150390609375 0.7781680046874999 0.3115234375 0.7233237156250001 0.2421874984375 0.671583821875 0.2441406234375 0.6322615046875 0.1816406234375 0.5391296921875 0.154296875 0.510155353125 0.1914062515625 0.4925637890625 0.1689453109375 0.46462424843749994 0.2304687484375 0.41702354531249997 0.173828125 0.354935675 0.1660156234375 0.32078734375 0.1982421859375 0.31354375624999997 0.236328125 0.25042108750000003 0.2412109390625 0.215237959375 0.3212890609375 0.17177645 0.3544921859375 0.16763725625 0.4384765625 0.12417574375 0.5263671859375 0.157289275 0.7412109390625 0.19040280781250002 0.8037109390625 0.32389173437499996 0.9140625015625 0.39632758749999997 0.939453125 0.45013707812500003 0.9296874984375 0.487389803125 0.9414062515625 0.5246425234375 0.9257812515625 0.5877651921875 0.9404296890625 0.71711493125 0.8984374984375 0.846464665625 0.7412109390625 0.8568126421875 0.5419921859375 0.8361166875 0.4638671859375


================================================
FILE: TumorDetection/valid/labels/no_tumor_951_jpg.rf.1176685798923047beda3f04c8486b0c.txt
================================================
0 0.9646920484375 0.5361328109375 0.9600652328124999 0.4033203109375 0.867528825 0.2021484390625 0.8108502718749999 0.140625 0.7090602234375 0.078125 0.58876289375 0.046875 0.373615746875 0.06445312343750001 0.2510050046875 0.11132812343750001 0.13880461093750002 0.2041015609375 0.0323877421875 0.3857421890625 0.0555218453125 0.5908203109375 0.10179004843749999 0.7431640609375 0.166565534375 0.8349609390625 0.246378184375 0.9042968765625 0.3435414125 0.953125 0.49275636875 0.9697265609375 0.62809086875 0.95703125 0.786559465625 0.8876953109375 0.867528825 0.7880859390625 0.9646920484375 0.5361328109375


================================================
FILE: TumorDetection/valid/labels/no_tumor_956_jpg.rf.320de10ee7815a41f6c73642dc7bc722.txt
================================================
0 0.760047934375 0.5537109375 0.7091168875 0.4287109375 0.6973635687499999 0.3115234375 0.6542680671875 0.2431640625 0.6062753515625 0.2050781234375 0.5377143249999999 0.1816406234375 0.5181254609375 0.18359375 0.50441325625 0.2011718765625 0.463276640625 0.1816406234375 0.43193445781250006 0.1816406234375 0.353579 0.21875 0.2918740765625 0.3095703125 0.28795630625 0.3759765625 0.2683674390625 0.4052734375 0.2507374625 0.4912109375 0.23310748750000002 0.5244140625 0.2272308265625 0.6357421875 0.26249078125 0.7275390625 0.3379079078125 0.8203125 0.406468934375 0.859375 0.453482209375 0.8652343765625 0.488742165625 0.8515625 0.4985365953125 0.8691406234375 0.5484882 0.8798828125 0.6552475109375 0.8300781234375 0.7267468671875 0.7353515625 0.7639657078125001 0.6298828125 0.760047934375 0.5537109375
0 0.6895280234375 0.2451171875 0.6523091796875 0.2412109375 0.6660213859375 0.2490234375 0.6895280234375 0.2451171875


================================================
FILE: TumorDetection/valid/labels/no_tumor_957_jpg.rf.449e467036fb02ccb06154af2e28e7f7.txt
================================================
0 0.8970566890624999 0.6669921875 0.903433865625 0.4794921875 0.837536334375 0.2763671875 0.7386900421875 0.1523437515625 0.6111464375 0.08984375156249999 0.436836846875 0.08203124843750001 0.319921875 0.1289062484375 0.20619549375 0.2568359359375 0.148800875 0.4228515640625 0.148800875 0.6201171875 0.21682412968749998 0.7685546875 0.319921875 0.8710937515625 0.4474654796875 0.9179687515625 0.5760719484375 0.9189453125 0.6727925156250001 0.9042968765625 0.7386900421875 0.8730468765625 0.846039246875 0.7763671875 0.8970566890624999 0.6669921875


================================================
FILE: TumorDetection/valid/labels/no_tumor_966_jpg.rf.46c7980a932d6269c724c002a73fe69a.txt
================================================
0 0.734375 0.8720703125 0.771484375 0.8251953125 0.8007812515625 0.7685546875 0.833984375 0.6064453125 0.833984375 0.4951171875 0.810546875 0.3564453125 0.763671875 0.2158203125 0.736328125 0.1689453125 0.6806640625 0.119140625 0.6298828125 0.08984374843750001 0.5791015625 0.07421874843750001 0.4619140625 0.07421874843750001 0.3955078125 0.0859375 0.3544921875 0.10546874843750001 0.279296875 0.1669921875 0.251953125 0.2001953125 0.2148437484375 0.2880859375 0.1679687484375 0.5166015625 0.166015625 0.6142578125 0.181640625 0.7060546875 0.212890625 0.8037109375 0.2382812515625 0.8466796875 0.2783203125 0.8945312515625 0.3583984375 0.935546875 0.4189453125 0.9570312515625 0.4921875 0.9599609375 0.5849609375 0.951171875 0.6181640625 0.9414062515625 0.7080078125 0.8945312515625 0.734375 0.8720703125


================================================
FILE: TumorDetection/valid/labels/no_tumor_969_jpg.rf.c177054ee4015cf9d5171c5618e3ef6d.txt
================================================
0 0.8425649578125001 0.6650390640625 0.8893741234375 0.5126953125 0.8477659781250001 0.4013671875 0.8165598656249999 0.3662109359375 0.782753246875 0.2763671875 0.735944084375 0.2392578125 0.720341028125 0.2138671875 0.6384249921875 0.181640625 0.5448066625 0.166015625 0.5110000453125 0.166015625 0.48759546093750006 0.1757812484375 0.44858782656249996 0.162109375 0.39397713125 0.1875 0.3445674609375 0.189453125 0.300358803125 0.205078125 0.2743537140625 0.205078125 0.2600509125 0.2158203125 0.22884480312500002 0.2724609359375 0.19763869375 0.2939453125 0.1846361484375 0.3447265640625 0.132625965625 0.4150390640625 0.117022909375 0.4970703125 0.119623421875 0.5595703125 0.1534300390625 0.6494140640625 0.1846361484375 0.6767578125 0.189837165625 0.7001953125 0.22624429531249998 0.7294921875 0.2561501484375 0.7734375 0.2951577859375 0.794921875 0.349768478125 0.8085937515625 0.40437916874999996 0.8398437515625 0.5825140453125 0.8408203125 0.6540280453125 0.828125 0.74244535625 0.78125 0.76715019375 0.7587890640625 0.8425649578125001 0.6650390640625


================================================
FILE: TumorDetection/valid/labels/no_tumor_995_jpg.rf.b3de5dbbc16cdd0204f064fe483c1c37.txt
================================================
0 0.814903846875 0.7998046875 0.8533653843750001 0.7646484375 0.877403846875 0.6357421875 0.8605769234375 0.5712890625 0.8581730765625 0.5126953125 0.8389423078125 0.4716796875 0.8269230765625 0.4013671875 0.7956730765625 0.3349609375 0.7956730765625 0.3037109375 0.7331730765625 0.2294921875 0.6766826921875 0.189453125 0.618990384375 0.1796875 0.5925480765625 0.162109375 0.5540865390625 0.1640625 0.513221153125 0.181640625 0.48677884687499995 0.162109375 0.44831730781249995 0.16015625 0.3233173078125 0.19921875 0.23798076875000002 0.2958984375 0.23317307656250003 0.3369140625 0.221153846875 0.3525390625 0.1730769234375 0.5029296875 0.1634615390625 0.6259765625 0.16826923124999998 0.6923828125 0.23076923124999998 0.8154296875 0.290865384375 0.8740234375 0.2740384609375 0.8857421875 0.3209134609375 0.919921875 0.34014423125 0.9296875 0.37139423125 0.92578125 0.4459134609375 0.9453125 0.47956730781249995 0.94140625 0.5204326921875 0.921875 0.546875 0.943359375 0.60576923125 0.9443359375 0.6334134609375 0.94140625 0.734375 0.90625 0.7980769234375 0.8466796875 0.814903846875 0.7998046875


================================================
FILE: TumorDetection/valid/labels/no_tumor_996_jpg.rf.408890d2c8475c8c401a82afec20f035.txt
================================================
0 0.732421875 0.8701171875 0.79296875 0.7978515625 0.80859375 0.7763671875 0.82421875 0.7451171875 0.83203125 0.7197265625 0.85546875 0.6044921875 0.859375 0.5537109375 0.85546875 0.4951171875 0.849609375 0.4658203125 0.8203125 0.3603515625 0.80859375 0.3037109375 0.78515625 0.2333984375 0.755859375 0.1708984375 0.6943359375 0.1015625 0.6455078125 0.06640625 0.6201171875 0.0546875 0.5712890625 0.04296875 0.5087890625 0.03515625 0.4169921875 0.048828125 0.3505859375 0.07421875 0.3095703125 0.10546875 0.26171875 0.1591796875 0.21484375 0.2548828125 0.197265625 0.3056640625 0.18359375 0.3759765625 0.15234375 0.4970703125 0.146484375 0.5751953125 0.15625 0.6806640625 0.17578125 0.7529296875 0.19921875 0.7919921875 0.2861328125 0.8828125 0.3349609375 0.916015625 0.3876953125 0.94140625 0.4130859375 0.94921875 0.4599609375 0.95703125 0.5546875 0.9580078125 0.5849609375 0.955078125 0.6201171875 0.9453125 0.6787109375 0.916015625 0.732421875 0.8701171875


================================================
FILE: TumorDetection/valid/labels/no_tumor_999_jpg.rf.d1687a97dc2837381fbe3a7371f3c99c.txt
================================================
0 0.795757575 0.7568359375 0.8154545453125 0.7138671875 0.829242425 0.6357421875 0.8371212124999999 0.5634765625 0.8371212124999999 0.4951171875 0.8272727265625001 0.3896484375 0.8154545453125 0.3427734375 0.79181818125 0.2744140625 0.7583333328124999 0.2060546875 0.7327272734375 0.1708984375 0.67856060625 0.119140625 0.6371969703125 0.091796875 0.603712121875 0.076171875 0.546590909375 0.068359375 0.4658333328125 0.06640625 0.432348484375 0.0703125 0.371287878125 0.09765625 0.3121969703125 0.13671875 0.275757575 0.1787109375 0.200909090625 0.3251953125 0.1693939390625 0.4130859375 0.1556060609375 0.4736328125 0.1496969703125 0.5888671875 0.1556060609375 0.6494140625 0.1693939390625 0.7080078125 0.193030303125 0.7509765625 0.210757575 0.7978515625 0.290530303125 0.88671875 0.34568181875000004 0.927734375 0.40083333281250005 0.94921875 0.477651515625 0.958984375 0.533787878125 0.9580078125 0.6017424250000001 0.94140625 0.67068181875 0.8984375 0.720909090625 0.8525390625 0.795757575 0.7568359375


================================================
FILE: TumorDetection/valid/labels/pituitary_1009_jpg.rf.e179a6d77c8a0ce652c26b7aee667ceb.txt
================================================
3 0.5166015625 0.3984375 0.4912109375 0.40625 0.46484375 0.4423828125 0.462890625 0.4560546875 0.478515625 0.4580078125 0.5302734375 0.44921875 0.537109375 0.4326171875 0.533203125 0.4111328125 0.5166015625 0.3984375


================================================
FILE: TumorDetection/valid/labels/pituitary_1026_jpg.rf.fa615e9c25e6aa3a263f679407b4a1a5.txt
================================================
3 0.5625 0.4912109375 0.5546875 0.4736328125 0.5263671875 0.455078125 0.5146484375 0.46484375 0.4970703125 0.46484375 0.4951171875 0.45703125 0.4755859375 0.45703125 0.4736328125 0.4453125 0.4560546875 0.443359375 0.443359375 0.4521484375 0.421875 0.4892578125 0.421875 0.5166015625 0.4375 0.5341796875 0.4453125 0.5615234375 0.4384765625 0.5625 0.4365234375 0.5546875 0.431640625 0.5595703125 0.4658203125 0.58984375 0.51171875 0.5908203125 0.525390625 0.5810546875 0.541015625 0.5478515625 0.560546875 0.5322265625 0.5625 0.4912109375


================================================
FILE: TumorDetection/valid/labels/pituitary_1044_jpg.rf.19d7eda33009adc9b1db6bd9083d8b98.txt
================================================
3 0.556640625 0.3505859375 0.5244140625 0.326171875 0.4833984375 0.322265625 0.4365234375 0.337890625 0.431640625 0.3427734375 0.4404296875 0.341796875 0.44140625 0.3486328125 0.41796875 0.3623046875 0.4248046875 0.373046875 0.4296875 0.3701171875 0.4375 0.3955078125 0.4306640625 0.41015625 0.421875 0.4052734375 0.423828125 0.3916015625 0.416015625 0.3935546875 0.423828125 0.4443359375 0.4326171875 0.451171875 0.486328125 0.4560546875 0.5400390625 0.447265625 0.5625 0.4189453125 0.564453125 0.3779296875 0.556640625 0.3505859375


================================================
FILE: TumorDetection/valid/labels/pituitary_1067_jpg.rf.0c9002bc461810e986834b904ca24f6e.txt
================================================
3 0.47265625 0.3212890625 0.47265625 0.3603515625 0.462890625 0.3818359375 0.4736328125 0.3984375 0.4931640625 0.400390625 0.5009765625 0.4140625 0.533203125 0.4169921875 0.572265625 0.3701171875 0.568359375 0.3115234375 0.5380859375 0.29296875 0.47265625 0.3212890625


================================================
FILE: TumorDetection/valid/labels/pituitary_1076_jpg.rf.6a8c3f73b891ffe6f81cf8e03582dafc.txt
================================================
3 0.546875 0.4560546875 0.5126953125 0.42578125 0.4560546875 0.42578125 0.419921875 0.4677734375 0.419921875 0.4951171875 0.4375 0.5302734375 0.427734375 0.5576171875 0.439453125 0.5654296875 0.416015625 0.5810546875 0.431640625 0.5908203125 0.4091796875 0.59375 0.4052734375 0.583984375 0.38671875 0.6220703125 0.4013671875 0.62890625 0.4404296875 0.587890625 0.4658203125 0.580078125 0.5126953125 0.599609375 0.5458984375 0.59375 0.587890625 0.6318359375 0.58203125 0.5791015625 0.5546875 0.5263671875 0.556640625 0.4853515625 0.546875 0.4560546875


================================================
FILE: TumorDetection/valid/labels/pituitary_1092_jpg.rf.1d6a0ba11712209997009f547c16d238.txt
================================================
3 0.5478515625 0.46875 0.5283203125 0.462890625 0.4677734375 0.45703125 0.4345703125 0.46484375 0.41015625 0.5048828125 0.4140625 0.5517578125 0.4521484375 0.6015625 0.4755859375 0.609375 0.521484375 0.6083984375 0.537109375 0.5966796875 0.5703125 0.5458984375 0.564453125 0.4853515625 0.5478515625 0.46875


================================================
FILE: TumorDetection/valid/labels/pituitary_1098_jpg.rf.d91b94211cbec86e864f47fd77a81eb3.txt
================================================
3 0.51171875 0.5283203125 0.53515625 0.5263671875 0.541015625 0.4736328125 0.5146484375 0.43359375 0.4755859375 0.4296875 0.443359375 0.4462890625 0.431640625 0.4912109375 0.45703125 0.5283203125 0.46484375 0.5537109375 0.498046875 0.5576171875 0.51953125 0.5498046875 0.4912109375 0.5390625 0.5263671875 0.53515625 0.51171875 0.5283203125


================================================
FILE: TumorDetection/valid/labels/pituitary_1104_jpg.rf.77a4b9e4245dc05bc100a23e1cd0acb9.txt
================================================
3 0.5830078125 0.373046875 0.4755859375 0.373046875 0.431640625 0.4072265625 0.423828125 0.4287109375 0.4306640625 0.482421875 0.56640625 0.4853515625 0.6044921875 0.482421875 0.6171875 0.4619140625 0.6171875 0.4267578125 0.5830078125 0.373046875


================================================
FILE: TumorDetection/valid/labels/pituitary_1107_jpg.rf.f59ef80dcd2a4136de36124616bc435c.txt
================================================
3 0.5126953125 0.34765625 0.4697265625 0.349609375 0.45703125 0.3603515625 0.44921875 0.3837890625 0.458984375 0.4326171875 0.4853515625 0.4609375 0.5234375 0.4638671875 0.552734375 0.4345703125 0.552734375 0.3681640625 0.5439453125 0.357421875 0.5126953125 0.34765625


================================================
FILE: TumorDetection/valid/labels/pituitary_1108_jpg.rf.a7bf774d2fb15ba62f47d79f33b74a24.txt
================================================
3 0.5224609375 0.5 0.4736328125 0.501953125 0.4462890625 0.513671875 0.431640625 0.5341796875 0.43359375 0.5732421875 0.4453125 0.5966796875 0.4775390625 0.630859375 0.529296875 0.6298828125 0.57421875 0.5654296875 0.57421875 0.5439453125 0.55859375 0.5146484375 0.5224609375 0.5


================================================
FILE: TumorDetection/valid/labels/pituitary_1121_jpg.rf.b4704ff58bc42db83bf464df9c19e75b.txt
================================================
3 0.5556640625 0.50390625 0.55859375 0.4990234375 0.556640625 0.4833984375 0.5380859375 0.478515625 0.53125 0.4677734375 0.54296875 0.4501953125 0.54296875 0.4345703125 0.5166015625 0.4140625 0.4970703125 0.416015625 0.48046875 0.4287109375 0.462890625 0.4716796875 0.443359375 0.4931640625 0.4580078125 0.53515625 0.509765625 0.5458984375 0.552734375 0.5048828125 0.5439453125 0.484375 0.5537109375 0.484375 0.5556640625 0.50390625


================================================
FILE: TumorDetection/valid/labels/pituitary_1127_jpg.rf.aeffce333eb2c5de56ffdb1d587c0369.txt
================================================
3 0.4345703125 0.541015625 0.408203125 0.5771484375 0.4130859375 0.58984375 0.42578125 0.5908203125 0.443359375 0.5751953125 0.443359375 0.5654296875 0.43359375 0.5634765625 0.4345703125 0.541015625
3 0.60546875 0.5087890625 0.5888671875 0.498046875 0.5576171875 0.494140625 0.5166015625 0.4609375 0.4775390625 0.470703125 0.462890625 0.4853515625 0.45703125 0.5185546875 0.4375 0.5478515625 0.4560546875 0.548828125 0.4677734375 0.564453125 0.5009765625 0.580078125 0.55859375 0.5830078125 0.5791015625 0.564453125 0.5947265625 0.576171875 0.626953125 0.5712890625 0.60546875 0.5087890625


================================================
FILE: TumorDetection/valid/labels/pituitary_1152_jpg.rf.e47bd8fa3a9c1e4f829b69cb4375d7d8.txt
================================================
3 0.54296875 0.4267578125 0.533203125 0.3994140625 0.5166015625 0.3828125 0.4970703125 0.384765625 0.478515625 0.4091796875 0.4765625 0.4775390625 0.484375 0.5068359375 0.4951171875 0.51953125 0.5205078125 0.517578125 0.552734375 0.5322265625 0.568359375 0.5205078125 0.568359375 0.4833984375 0.54296875 0.4599609375 0.54296875 0.4267578125


================================================
FILE: TumorDetection/valid/labels/pituitary_1165_jpg.rf.94e81c2cb82675782dc4e5921be405ab.txt
================================================
3 0.5615234375 0.59375 0.5263671875 0.58984375 0.5029296875 0.578125 0.4326171875 0.576171875 0.408203125 0.6064453125 0.408203125 0.6455078125 0.4287109375 0.658203125 0.4462890625 0.65234375 0.48828125 0.6572265625 0.5185546875 0.640625 0.5712890625 0.646484375 0.583984375 0.6220703125 0.5615234375 0.59375


================================================
FILE: TumorDetection/valid/labels/pituitary_1171_jpg.rf.c6eef3d6c3ec7831fbcf0fc0aca4fb3b.txt
================================================
3 0.548828125 0.5732421875 0.5283203125 0.5546875 0.5107421875 0.55078125 0.4853515625 0.556640625 0.46875 0.5849609375 0.47265625 0.6318359375 0.462890625 0.6572265625 0.49609375 0.6806640625 0.5263671875 0.67578125 0.568359375 0.6416015625 0.552734375 0.6103515625 0.548828125 0.5732421875


================================================
FILE: TumorDetection/valid/labels/pituitary_1180_jpg.rf.ae6aa98e15f082e177ac2090c74ef02b.txt
================================================
3 0.615234375 0.5810546875 0.60546875 0.5380859375 0.5859375 0.5146484375 0.568359375 0.4736328125 0.5400390625 0.4453125 0.5048828125 0.4296875 0.4755859375 0.43359375 0.44921875 0.4560546875 0.4375 0.4794921875 0.439453125 0.5361328125 0.41796875 0.5576171875 0.4248046875 0.5625 0.4345703125 0.556640625 0.439453125 0.5673828125 0.4248046875 0.572265625 0.4169921875 0.55859375 0.4140625 0.5634765625 0.42578125 0.5908203125 0.4482421875 0.61328125 0.48046875 0.6220703125 0.5302734375 0.61328125 0.5947265625 0.619140625 0.60546875 0.6123046875 0.615234375 0.5810546875


================================================
FILE: TumorDetection/valid/labels/pituitary_1197_jpg.rf.07965526c920bba41c1d5b7dd0089f1f.txt
================================================
3 0.546875 0.4775390625 0.5283203125 0.458984375 0.5048828125 0.4453125 0.4677734375 0.4453125 0.447265625 0.4736328125 0.451171875 0.5361328125 0.427734375 0.5615234375 0.4375 0.5771484375 0.4189453125 0.572265625 0.416015625 0.5810546875 0.4326171875 0.59375 0.5625 0.6064453125 0.6142578125 0.587890625 0.609375 0.5712890625 0.576171875 0.5400390625 0.546875 0.4775390625


================================================
FILE: TumorDetection/valid/labels/pituitary_1198_jpg.rf.11d1c54285218a5b1dde50d508cfc794.txt
================================================
3 0.572265625 0.4814453125 0.5517578125 0.4453125 0.5244140625 0.443359375 0.4892578125 0.453125 0.4521484375 0.490234375 0.4248046875 0.498046875 0.4140625 0.5146484375 0.412109375 0.5712890625 0.505859375 0.6162109375 0.5361328125 0.603515625 0.564453125 0.5751953125 0.564453125 0.5126953125 0.572265625 0.4814453125


================================================
FILE: TumorDetection/valid/labels/pituitary_1208_jpg.rf.f6af8616dfe73e7f284d9db3de136782.txt
================================================
3 0.5146484375 0.505859375 0.4677734375 0.5078125 0.412109375 0.5400390625 0.4169921875 0.5546875 0.4521484375 0.576171875 0.462890625 0.5751953125 0.5283203125 0.56640625 0.544921875 0.5439453125 0.5390625 0.5205078125 0.5146484375 0.505859375


================================================
FILE: TumorDetection/valid/labels/pituitary_1222_jpg.rf.9560cfc6b55c48f9aa1692716de4e245.txt
================================================
3 0.4984375 0.3296875 0.23046875 0.14375
3 0.5171595984375 0.388671875 0.5386439734375 0.388671875 0.5532924109375 0.3720703125 0.5708705359375 0.3232421875 0.5904017859375 0.2998046875 0.5884486609375 0.2880859375 0.5738002234375 0.27734375 0.5581752234375 0.279296875 0.49372209843750003 0.318359375 0.47809709843750003 0.3046875 0.40387834843750003 0.2734375 0.39704241093750003 0.2822265625 0.40094866093750003 0.3056640625 0.43024553593750003 0.3447265625 0.44587053593750003 0.3974609375 0.46149553593750003 0.4013671875 0.48688616093750003 0.3720703125 0.49372209843750003 0.3515625 0.49665178593750003 0.3681640625 0.5171595984375 0.388671875


================================================
FILE: TumorDetection/valid/labels/pituitary_1247_jpg.rf.0f885c770dfeadef6da09d14a0aec1b9.txt
================================================
3 0.607421875 0.5595703125 0.599609375 0.5400390625 0.572265625 0.5185546875 0.55859375 0.4677734375 0.5283203125 0.458984375 0.5166015625 0.4453125 0.4130859375 0.474609375 0.39453125 0.5341796875 0.3642578125 0.537109375 0.359375 0.5498046875 0.34375 0.5595703125 0.349609375 0.5830078125 0.3740234375 0.599609375 0.4169921875 0.587890625 0.4326171875 0.603515625 0.466796875 0.6083984375 0.5146484375 0.603515625 0.5302734375 0.576171875 0.5654296875 0.59765625 0.607421875 0.5595703125


================================================
FILE: TumorDetection/valid/labels/pituitary_1260_jpg.rf.962d995dae3c5e32c701d1b778f74d41.txt
================================================
3 0.595703125 0.5771484375 0.5693359375 0.55859375 0.544921875 0.5654296875 0.560546875 0.5263671875 0.548828125 0.4873046875 0.5263671875 0.4609375 0.5009765625 0.451171875 0.4599609375 0.46875 0.421875 0.5107421875 0.421875 0.5478515625 0.4453125 0.5732421875 0.4287109375 0.57421875 0.40625 0.5947265625 0.400390625 0.6279296875 0.4091796875 0.642578125 0.4775390625 0.681640625 0.513671875 0.6826171875 0.5966796875 0.66015625 0.60546875 0.6494140625 0.60546875 0.6298828125 0.595703125 0.5771484375


================================================
FILE: TumorDetection/valid/labels/pituitary_1262_jpg.rf.467185be0f99bf9b3d421767f440bf2b.txt
================================================
3 0.63671875 0.6357421875 0.595703125 0.5654296875 0.5625 0.5517578125 0.5703125 0.5244140625 0.5595703125 0.509765625 0.5263671875 0.48828125 0.4736328125 0.486328125 0.462890625 0.5029296875 0.4609375 0.5263671875 0.474609375 0.5654296875 0.4248046875 0.583984375 0.404296875 0.6240234375 0.43359375 0.6396484375 0.4658203125 0.69140625 0.513671875 0.6943359375 0.5302734375 0.689453125 0.5576171875 0.662109375 0.6240234375 0.650390625 0.63671875 0.6357421875


================================================
FILE: TumorDetection/valid/labels/pituitary_1270_jpg.rf.625c288b1b6b7e19298316b10041e618.txt
================================================
3 0.537109375 0.5908203125 0.54296875 0.5615234375 0.5224609375 0.5390625 0.5068359375 0.533203125 0.46875 0.5478515625 0.46484375 0.5693359375 0.470703125 0.5810546875 0.45703125 0.5908203125 0.486328125 0.6298828125 0.466796875 0.6513671875 0.46484375 0.6708984375 0.4775390625 0.685546875 0.49609375 0.6845703125 0.5341796875 0.677734375 0.544921875 0.6630859375 0.544921875 0.6123046875 0.537109375 0.5908203125


================================================
FILE: TumorDetection/valid/labels/pituitary_1293_jpg.rf.227ec5924fa283fc25b48a0b5ddcd1f9.txt
================================================
3 0.5478515625 0.58984375 0.53125 0.6044921875 0.517578125 0.6435546875 0.5244140625 0.65625 0.541015625 0.6572265625 0.5693359375 0.64453125 0.59375 0.6162109375 0.5732421875 0.59375 0.5478515625 0.58984375


================================================
FILE: TumorDetection/valid/labels/pituitary_1295_jpg.rf.a63887af4973fc134e57b1af4e60c675.txt
================================================
3 0.515625 0.6474609375 0.5126953125 0.640625 0.546875 0.6259765625 0.552734375 0.5771484375 0.568359375 0.5498046875 0.568359375 0.4931640625 0.5361328125 0.470703125 0.4560546875 0.48046875 0.41796875 0.5205078125 0.4296875 0.5439453125 0.42578125 0.5576171875 0.4453125 0.5751953125 0.44921875 0.6357421875 0.4921875 0.6552734375 0.4970703125 0.6484375 0.4970703125 0.654296875 0.5068359375 0.654296875 0.515625 0.6474609375


================================================
FILE: TumorDetection/valid/labels/pituitary_1315_jpg.rf.0aebcf4347718666c98f20beea90aa41.txt
================================================
3 0.6220703125 0.5078125 0.5693359375 0.5234375 0.552734375 0.5380859375 0.544921875 0.5576171875 0.5791015625 0.595703125 0.609375 0.5966796875 0.6318359375 0.58984375 0.650390625 0.5634765625 0.6328125 0.5146484375 0.6220703125 0.5078125


================================================
FILE: TumorDetection/valid/labels/pituitary_1330_jpg.rf.7d13fca1123687a5db40fceeb350552a.txt
================================================
3 0.6025390625 0.5390625 0.5615234375 0.53515625 0.537109375 0.5595703125 0.53515625 0.5927734375 0.548828125 0.6142578125 0.55078125 0.6376953125 0.5859375 0.6533203125 0.6025390625 0.6484375 0.62109375 0.6220703125 0.6171875 0.5595703125 0.6025390625 0.5390625


================================================
FILE: TumorDetection/valid/labels/pituitary_1345_jpg.rf.537cc53b8da80358a6a661da0d61d3d5.txt
================================================
3 0.69140625 0.5576171875 0.650390625 0.5400390625 0.666015625 0.5048828125 0.6484375 0.4755859375 0.646484375 0.4521484375 0.6259765625 0.435546875 0.5986328125 0.43359375 0.5595703125 0.46484375 0.51171875 0.4853515625 0.490234375 0.5107421875 0.49609375 0.5302734375 0.490234375 0.5537109375 0.5009765625 0.564453125 0.5205078125 0.56640625 0.541015625 0.5888671875 0.544921875 0.6494140625 0.556640625 0.6689453125 0.5849609375 0.69140625 0.615234375 0.6982421875 0.67578125 0.6689453125 0.701171875 0.6181640625 0.701171875 0.5966796875 0.69140625 0.5576171875


================================================
FILE: TumorDetection/valid/labels/pituitary_1350_jpg.rf.98e44dd4b49fc352d659f065614ad9a0.txt
================================================
3 0.705078125 0.6416015625 0.703125 0.6083984375 0.689453125 0.5751953125 0.658203125 0.5576171875 0.681640625 0.5185546875 0.673828125 0.4638671875 0.6376953125 0.419921875 0.5849609375 0.4140625 0.513671875 0.4462890625 0.501953125 0.5048828125 0.501953125 0.5498046875 0.5126953125 0.576171875 0.5400390625 0.583984375 0.5546875 0.5966796875 0.560546875 0.6162109375 0.55859375 0.6787109375 0.5986328125 0.693359375 0.623046875 0.6923828125 0.6455078125 0.69140625 0.6708984375 0.66796875 0.6953125 0.6572265625 0.705078125 0.6416015625


================================================
FILE: TumorDetection/valid/labels/pituitary_1351_jpg.rf.ee8fe8c699b2b99e01c7f41d69410e98.txt
================================================
3 0.66015625 0.5361328125 0.67578125 0.4970703125 0.6474609375 0.44140625 0.6162109375 0.42578125 0.5830078125 0.42578125 0.5537109375 0.43359375 0.521484375 0.4580078125 0.51171875 0.5048828125 0.5341796875 0.560546875 0.5673828125 0.580078125 0.603515625 0.5810546875 0.619140625 0.5712890625 0.6240234375 0.546875 0.6435546875 0.55078125 0.66015625 0.5361328125


================================================
FILE: TumorDetection/valid/labels/pituitary_1376_jpg.rf.36043ee2f3fc026f1ae2062299c0b505.txt
================================================
3 0.6201171875 0.46875 0.5830078125 0.474609375 0.5625 0.4951171875 0.568359375 0.5361328125 0.595703125 0.5380859375 0.6220703125 0.529296875 0.634765625 0.5126953125 0.63671875 0.4833984375 0.6201171875 0.46875


================================================
FILE: TumorDetection/valid/labels/pituitary_1380_jpg.rf.d2cddf63eb15ae086dba6d9296236f71.txt
================================================
3 0.67578125 0.5068359375 0.6396484375 0.47265625 0.6201171875 0.4609375 0.5986328125 0.4609375 0.5791015625 0.470703125 0.560546875 0.4951171875 0.576171875 0.5205078125 0.572265625 0.5712890625 0.5849609375 0.5859375 0.626953125 0.5869140625 0.6591796875 0.578125 0.68359375 0.5537109375 0.6875 0.5380859375 0.67578125 0.5068359375


================================================
FILE: TumorDetection/valid/labels/pituitary_1386_jpg.rf.b4cdf1145d81200512ff6d610473eceb.txt
================================================
3 0.5576171875 0.560546875 0.509765625 0.6025390625 0.5078125 0.6552734375 0.587890625 0.6826171875 0.625 0.6630859375 0.630859375 0.6044921875 0.6005859375 0.56640625 0.5576171875 0.560546875


================================================
FILE: TumorDetection/valid/labels/pituitary_139_jpg.rf.6b793346b7668ed1f42c419b54999916.txt
================================================
3 0.4677734375 0.5234375 0.4482421875 0.5078125 0.4150390625 0.498046875 0.3662109375 0.513671875 0.3515625 0.5419921875 0.361328125 0.5751953125 0.3740234375 0.58984375 0.40625 0.5947265625 0.4599609375 0.5859375 0.4765625 0.5732421875 0.484375 0.5556640625 0.4677734375 0.5234375


================================================
FILE: TumorDetection/valid/labels/pituitary_1437_jpg.rf.f58b5bec741ad04c78c44c2c5e2c430a.txt
================================================
3 0.6279296875 0.537109375 0.6044921875 0.53125 0.5908203125 0.533203125 0.57421875 0.5517578125 0.572265625 0.5732421875 0.546875 0.6103515625 0.546875 0.6396484375 0.5625 0.6767578125 0.5751953125 0.6875 0.599609375 0.6904296875 0.6337890625 0.677734375 0.650390625 0.6494140625 0.6484375 0.5556640625 0.6279296875 0.537109375


================================================
FILE: TumorDetection/valid/labels/pituitary_1440_jpg.rf.914705ce7e047c1b93557f9724dd6a4e.txt
================================================
3 0.6142578125 0.578125 0.5888671875 0.5703125 0.560546875 0.5888671875 0.55859375 0.6318359375 0.580078125 0.6728515625 0.607421875 0.6826171875 0.634765625 0.6435546875 0.62890625 0.6220703125 0.642578125 0.6005859375 0.6142578125 0.578125


================================================
FILE: TumorDetection/valid/labels/pituitary_1448_jpg.rf.11d6d3135dfcef91d320133d2b2fd2e3.txt
================================================
3 0.56640625 0.5458984375 0.5498046875 0.521484375 0.4658203125 0.513671875 0.4453125 0.5263671875 0.427734375 0.5634765625 0.46875 0.6279296875 0.5 0.6435546875 0.5263671875 0.640625 0.544921875 0.6240234375 0.56640625 0.5869140625 0.56640625 0.5458984375


================================================
FILE: TumorDetection/valid/labels/pituitary_1449_jpg.rf.a391cc17ec28a458bf5d8e86a0ea1b8f.txt
================================================
3 0.5576171875 0.46484375 0.5107421875 0.470703125 0.4677734375 0.46484375 0.4462890625 0.474609375 0.43359375 0.4931640625 0.4296875 0.5400390625 0.4697265625 0.58203125 0.4951171875 0.599609375 0.5078125 0.5986328125 0.5439453125 0.5859375 0.580078125 0.5341796875 0.58203125 0.4873046875 0.5576171875 0.46484375


================================================
FILE: TumorDetection/valid/labels/pituitary_1456_jpg.rf.65fea9431b09018b4afdc9e26a3102ec.txt
================================================
3 0.47265625 0.5146484375 0.4931640625 0.5390625 0.55078125 0.5419921875 0.568359375 0.5146484375 0.5478515625 0.494140625 0.4990234375 0.494140625 0.47265625 0.5146484375


================================================
FILE: TumorDetection/valid/labels/pituitary_156_jpg.rf.169d34a242c1aab392535e0612a1fbd5.txt
================================================
3 0.533203125 0.5654296875 0.533203125 0.5517578125 0.5205078125 0.5390625 0.5068359375 0.52734375 0.4892578125 0.52734375 0.46484375 0.5498046875 0.4609375 0.5791015625 0.439453125 0.5986328125 0.4453125 0.6064453125 0.44140625 0.6318359375 0.462890625 0.6962890625 0.4814453125 0.7109375 0.5078125 0.7099609375 0.5390625 0.6845703125 0.568359375 0.6044921875 0.556640625 0.5810546875 0.533203125 0.5654296875


================================================
FILE: TumorDetection/valid/labels/pituitary_171_jpg.rf.6290ddd1131aafcb4bce1167db5f19ea.txt
================================================
3 0.4150390625 0.53125 0.3720703125 0.544921875 0.33984375 0.5712890625 0.369140625 0.6181640625 0.392578125 0.6298828125 0.4267578125 0.625 0.44140625 0.5986328125 0.43359375 0.5517578125 0.4150390625 0.53125


================================================
FILE: TumorDetection/valid/labels/pituitary_189_jpg.rf.649a81cb433ea19e0f07aa8b00f2554d.txt
================================================
3 0.423828125 0.4853515625 0.431640625 0.5283203125 0.4580078125 0.5546875 0.490234375 0.5556640625 0.505859375 0.5048828125 0.4716796875 0.453125 0.4443359375 0.453125 0.423828125 0.4853515625


================================================
FILE: TumorDetection/valid/labels/pituitary_197_jpg.rf.39d697f7e73591c428e6546b826ac68f.txt
================================================
3 0.3935546875 0.55859375 0.3701171875 0.546875 0.36328125 0.5517578125 0.359375 0.5830078125 0.373046875 0.6064453125 0.38671875 0.6123046875 0.4169921875 0.6015625 0.4296875 0.5888671875 0.4267578125 0.546875 0.3935546875 0.55859375


================================================
FILE: TumorDetection/valid/labels/pituitary_207_jpg.rf.e268e6d86155a9c9017e1b816eea0eef.txt
================================================
3 0.4169921875 0.439453125 0.3876953125 0.4453125 0.3701171875 0.470703125 0.3505859375 0.46875 0.3359375 0.4814453125 0.3359375 0.5029296875 0.3671875 0.5146484375 0.4130859375 0.509765625 0.4375 0.4892578125 0.4375 0.4638671875 0.4169921875 0.439453125


================================================
FILE: TumorDetection/valid/labels/pituitary_214_jpg.rf.b4ada692026dbea30c65e447f3430d69.txt
================================================
3 0.4853515625 0.37109375 0.4462890625 0.375 0.439453125 0.3837890625 0.44140625 0.4423828125 0.4599609375 0.462890625 0.482421875 0.4638671875 0.517578125 0.4462890625 0.5234375 0.4345703125 0.521484375 0.3935546875 0.4853515625 0.37109375


================================================
FILE: TumorDetection/valid/labels/pituitary_259_jpg.rf.e7a8af82dc71e9fcc2edb0a15ec476c4.txt
================================================
3 0.583984375 0.6201171875 0.568359375 0.5791015625 0.537109375 0.5595703125 0.560546875 0.5439453125 0.578125 0.5146484375 0.578125 0.4990234375 0.5517578125 0.482421875 0.55859375 0.4775390625 0.5498046875 0.470703125 0.5458984375 0.4765625 0.5341796875 0.470703125 0.5224609375 0.4765625 0.5185546875 0.46875 0.5146484375 0.474609375 0.4697265625 0.4765625 0.44921875 0.4970703125 0.453125 0.5263671875 0.47265625 0.5498046875 0.44140625 0.5634765625 0.421875 0.6025390625 0.41796875 0.6357421875 0.4443359375 0.6796875 0.490234375 0.6962890625 0.5458984375 0.6875 0.5712890625 0.671875 0.583984375 0.6513671875 0.583984375 0.6201171875


================================================
FILE: TumorDetection/valid/labels/pituitary_263_jpg.rf.915b29f4bfea50f135f1647859387a5f.txt
================================================
3 0.58203125 0.6025390625 0.568359375 0.5966796875 0.57421875 0.5908203125 0.56640625 0.5849609375 0.59765625 0.5498046875 0.611328125 0.4931640625 0.611328125 0.4716796875 0.59765625 0.4482421875 0.5380859375 0.41015625 0.5009765625 0.412109375 0.4775390625 0.42578125 0.421875 0.4931640625 0.4140625 0.5087890625 0.41796875 0.5576171875 0.404296875 0.5927734375 0.451171875 0.6845703125 0.4873046875 0.71484375 0.5234375 0.7216796875 0.544921875 0.7060546875 0.552734375 0.6572265625 0.58203125 0.6279296875 0.58203125 0.6025390625


================================================
FILE: TumorDetection/valid/labels/pituitary_288_jpg.rf.e802e126c7930bed0d6d3d7a0b83e900.txt
================================================
3 0.4462890625 0.447265625 0.427734375 0.4814453125 0.431640625 0.5029296875 0.4580078125 0.51953125 0.48828125 0.5224609375 0.509765625 0.4990234375 0.501953125 0.4560546875 0.4794921875 0.44140625 0.4462890625 0.447265625


================================================
FILE: TumorDetection/valid/labels/pituitary_311_jpg.rf.1204ad349fb21342fb37d4aaa6130976.txt
================================================
3 0.5107421875 0.56640625 0.4736328125 0.560546875 0.44140625 0.5986328125 0.44140625 0.6220703125 0.4501953125 0.630859375 0.52734375 0.6357421875 0.546875 0.6259765625 0.54296875 0.6123046875 0.556640625 0.5966796875 0.5107421875 0.56640625


================================================
FILE: TumorDetection/valid/labels/pituitary_338_jpg.rf.eeeb860df08888c687e8ffead7165ca1.txt
================================================
3 0.45703125 0.5068359375 0.4345703125 0.48828125 0.4091796875 0.482421875 0.3740234375 0.494140625 0.345703125 0.5244140625 0.3359375 0.5869140625 0.3681640625 0.65234375 0.3955078125 0.666015625 0.41796875 0.6650390625 0.4384765625 0.658203125 0.466796875 0.6279296875 0.47265625 0.5458984375 0.45703125 0.5068359375


================================================
FILE: TumorDetection/valid/labels/pituitary_34_jpg.rf.dd47d9c32ae228342cda345fd308c1ca.txt
================================================
3 0.478515625 0.4130859375 0.4970703125 0.427734375 0.51171875 0.4267578125 0.53515625 0.4150390625 0.541015625 0.3876953125 0.4775390625 0.3828125 0.478515625 0.4130859375


================================================
FILE: TumorDetection/valid/labels/pituitary_354_jpg.rf.9d4d3a32f68c95b22b9e370a0b4efa8d.txt
================================================
3 0.388671875 0.5771484375 0.390625 0.6064453125 0.431640625 0.6201171875 0.453125 0.6025390625 0.455078125 0.5693359375 0.4072265625 0.5546875 0.388671875 0.5771484375


================================================
FILE: TumorDetection/valid/labels/pituitary_36_jpg.rf.fd6705c944dab1fb3962524d35ddcf2f.txt
================================================
3 0.5615234375 0.509765625 0.4384765625 0.505859375 0.40625 0.5244140625 0.400390625 0.5498046875 0.40625 0.5888671875 0.478515625 0.6455078125 0.5263671875 0.634765625 0.591796875 0.5712890625 0.6015625 0.5498046875 0.5615234375 0.509765625


================================================
FILE: TumorDetection/valid/labels/pituitary_403_jpg.rf.d573650d6ff7a0154fc704d7607ef9a5.txt
================================================
3 0.564453125 0.5029296875 0.521484375 0.4697265625 0.517578125 0.4267578125 0.5087890625 0.41796875 0.4951171875 0.41796875 0.47265625 0.4326171875 0.4609375 0.4580078125 0.46484375 0.4814453125 0.435546875 0.5126953125 0.4296875 0.5849609375 0.4619140625 0.6171875 0.529296875 0.6240234375 0.5517578125 0.615234375 0.57421875 0.5869140625 0.572265625 0.5205078125 0.564453125 0.5029296875


================================================
FILE: TumorDetection/valid/labels/pituitary_404_jpg.rf.677d57ac4ee3c3fb96ce4efe22e9063f.txt
================================================
3 0.5244140625 0.669921875 0.51171875 0.6826171875 0.5009765625 0.712890625 0.4921875 0.7080078125 0.4755859375 0.671875 0.4462890625 0.677734375 0.44140625 0.7490234375 0.453125 0.7724609375 0.4736328125 0.787109375 0.5087890625 0.77734375 0.5224609375 0.791015625 0.53515625 0.7900390625 0.5625 0.7392578125 0.5546875 0.6748046875 0.5439453125 0.66796875 0.5244140625 0.669921875
3 0.552734375 0.4970703125 0.552734375 0.4736328125 0.5283203125 0.447265625 0.4951171875 0.439453125 0.4736328125 0.4453125 0.44140625 0.4697265625 0.41796875 0.5244140625 0.4140625 0.5654296875 0.4287109375 0.603515625 0.5234375 0.6142578125 0.5419921875 0.603515625 0.55859375 0.5791015625 0.5390625 0.5361328125 0.552734375 0.4970703125


================================================
FILE: TumorDetection/valid/labels/pituitary_411_jpg.rf.748049333a6fce2222cf7511835be3fb.txt
================================================
3 0.533203125 0.3154296875 0.5263671875 0.310546875 0.51953125 0.3173828125 0.5244140625 0.33984375 0.5087890625 0.32421875 0.4990234375 0.32421875 0.4765625 0.3564453125 0.4892578125 0.375 0.49609375 0.3740234375 0.5185546875 0.37109375 0.52734375 0.3603515625 0.5234375 0.3447265625 0.533203125 0.3154296875


================================================
FILE: TumorDetection/valid/labels/pituitary_470_jpg.rf.f1f9b1c6cfca246fb2d1a8cc708a417d.txt
================================================
3 0.5458984375 0.46484375 0.4853515625 0.458984375 0.48046875 0.4912109375 0.4609375 0.5048828125 0.4609375 0.5517578125 0.4814453125 0.576171875 0.5234375 0.5791015625 0.556640625 0.5556640625 0.564453125 0.5146484375 0.55078125 0.4970703125 0.5458984375 0.46484375


================================================
FILE: TumorDetection/valid/labels/pituitary_472_jpg.rf.d78e5d88f668fdb2d584e8cbc431c106.txt
================================================
3 0.4736328125 0.44921875 0.447265625 0.4697265625 0.451171875 0.4970703125 0.44140625 0.5068359375 0.44140625 0.5224609375 0.4775390625 0.568359375 0.498046875 0.5693359375 0.533203125 0.5556640625 0.552734375 0.5126953125 0.5029296875 0.45703125 0.4736328125 0.44921875


================================================
FILE: TumorDetection/valid/labels/pituitary_475_jpg.rf.81c6fdd17bdff50b43b76b75a97e3543.txt
================================================
3 0.5439453125 0.443359375 0.5263671875 0.431640625 0.5048828125 0.4296875 0.4970703125 0.44921875 0.443359375 0.4794921875 0.443359375 0.4931640625 0.453125 0.5166015625 0.4677734375 0.529296875 0.5009765625 0.54296875 0.51953125 0.5419921875 0.5361328125 0.537109375 0.5546875 0.5185546875 0.5625 0.4814453125 0.5439453125 0.443359375


================================================
FILE: TumorDetection/valid/labels/pituitary_493_jpg.rf.37401d79404f9906aaa99e85f7c88fe2.txt
================================================
3 0.4794921875 0.3984375 0.4638671875 0.404296875 0.435546875 0.4384765625 0.4501953125 0.466796875 0.494140625 0.4755859375 0.541015625 0.4580078125 0.54296875 0.4228515625 0.5146484375 0.400390625 0.4794921875 0.3984375


================================================
FILE: TumorDetection/valid/labels/pituitary_497_jpg.rf.aaef3a2853a5dd9e8309eca35811b527.txt
================================================
3 0.5263671875 0.396484375 0.5146484375 0.37890625 0.4501953125 0.400390625 0.443359375 0.4072265625 0.44921875 0.4384765625 0.4560546875 0.447265625 0.470703125 0.4462890625 0.5322265625 0.443359375 0.537109375 0.4072265625 0.544921875 0.4033203125 0.5263671875 0.396484375


================================================
FILE: TumorDetection/valid/labels/pituitary_533_jpg.rf.92c341f51cb080a7059811307a860796.txt
================================================
3 0.59375 0.4501953125 0.580078125 0.4306640625 0.5283203125 0.419921875 0.4912109375 0.435546875 0.48046875 0.4501953125 0.48046875 0.4755859375 0.49609375 0.5166015625 0.55859375 0.5224609375 0.611328125 0.5068359375 0.60546875 0.4775390625 0.591796875 0.4716796875 0.59375 0.4501953125


================================================
FILE: TumorDetection/valid/labels/pituitary_542_jpg.rf.c739efdaa53527636057bc1fcb68526c.txt
================================================
3 0.5537109375 0.486328125 0.4580078125 0.48828125 0.42578125 0.5224609375 0.421875 0.5478515625 0.427734375 0.5654296875 0.4677734375 0.60546875 0.4892578125 0.615234375 0.4970703125 0.630859375 0.517578125 0.6298828125 0.5859375 0.5673828125 0.591796875 0.5517578125 0.587890625 0.5263671875 0.5537109375 0.486328125


================================================
FILE: TumorDetection/valid/labels/pituitary_546_jpg.rf.72f5354a69759a3b37185b519178955d.txt
================================================
3 0.54296875 0.3681640625 0.5263671875 0.3515625 0.5126953125 0.349609375 0.4697265625 0.376953125 0.3984375 0.3916015625 0.3984375 0.4228515625 0.4111328125 0.439453125 0.4267578125 0.447265625 0.46484375 0.4462890625 0.5244140625 0.431640625 0.541015625 0.4189453125 0.546875 0.3837890625 0.54296875 0.3681640625


================================================
FILE: TumorDetection/valid/labels/pituitary_565_jpg.rf.d6cf0d552749c62f69b1cf1d3bc4d12c.txt
================================================
3 0.4951171875 0.375 0.4736328125 0.380859375 0.44921875 0.4072265625 0.44921875 0.4248046875 0.4658203125 0.44140625 0.560546875 0.4599609375 0.5625 0.4423828125 0.55078125 0.4052734375 0.5322265625 0.37890625 0.4951171875 0.375


================================================
FILE: TumorDetection/valid/labels/pituitary_567_jpg.rf.bb5b6ca6391cac363f1a1a69d33c07b5.txt
================================================
3 0.470703125 0.4130859375 0.46875 0.4443359375 0.4853515625 0.458984375 0.51171875 0.4619140625 0.53125 0.4462890625 0.533203125 0.4169921875 0.4814453125 0.400390625 0.470703125 0.4130859375


================================================
FILE: TumorDetection/valid/labels/pituitary_589_jpg.rf.ef5f8bc3b105320502868193ab191c87.txt
================================================
3 0.583984375 0.3837890625 0.583984375 0.3603515625 0.564453125 0.3310546875 0.5224609375 0.306640625 0.5009765625 0.3046875 0.48046875 0.3193359375 0.45703125 0.3623046875 0.431640625 0.4326171875 0.431640625 0.4599609375 0.4375 0.4814453125 0.4580078125 0.5 0.525390625 0.5029296875 0.5517578125 0.4921875 0.591796875 0.4521484375 0.59765625 0.4130859375 0.583984375 0.3837890625


================================================
FILE: TumorDetection/valid/labels/pituitary_60_jpg.rf.1a4b4517ce496f69b86facecb4296981.txt
================================================
3 0.4345703125 0.515625 0.41015625 0.5302734375 0.408203125 0.5947265625 0.44140625 0.6025390625 0.470703125 0.5830078125 0.4765625 0.5439453125 0.4619140625 0.5234375 0.4345703125 0.515625


================================================
FILE: TumorDetection/valid/labels/pituitary_626_jpg.rf.a1620fbfeae1bf27ab8d1e35ca5f321f.txt
================================================
3 0.560546875 0.4443359375 0.541015625 0.3818359375 0.4892578125 0.376953125 0.4521484375 0.392578125 0.44140625 0.4267578125 0.42578125 0.4365234375 0.42578125 0.4716796875 0.4140625 0.4970703125 0.40625 0.5634765625 0.4580078125 0.615234375 0.521484375 0.6181640625 0.57421875 0.5693359375 0.572265625 0.4990234375 0.556640625 0.4619140625 0.560546875 0.4443359375


================================================
FILE: TumorDetection/valid/labels/pituitary_639_jpg.rf.acc1152352e1d325b776d5728c807413.txt
================================================
3 0.5380859375 0.4921875 0.5029296875 0.486328125 0.4462890625 0.501953125 0.435546875 0.5244140625 0.4453125 0.5673828125 0.4755859375 0.6015625 0.51171875 0.6103515625 0.5400390625 0.6015625 0.56640625 0.5712890625 0.564453125 0.5087890625 0.5380859375 0.4921875


================================================
FILE: TumorDetection/valid/labels/pituitary_640_jpg.rf.e4c12aca7b7c313222602aebf557be0d.txt
================================================
3 0.46875 0.4736328125 0.4619140625 0.490234375 0.453125 0.4423828125 0.435546875 0.4404296875 0.4404296875 0.427734375 0.455078125 0.4404296875 0.462890625 0.4130859375 0.4560546875 0.40625 0.3876953125 0.40234375 0.3642578125 0.41015625 0.361328125 0.4189453125 0.37890625 0.4384765625 0.369140625 0.4443359375 0.357421875 0.4755859375 0.3515625 0.5419921875 0.3671875 0.5830078125 0.3916015625 0.599609375 0.408203125 0.5986328125 0.4072265625 0.59375 0.44921875 0.5830078125 0.462890625 0.5419921875 0.46875 0.4736328125


================================================
FILE: TumorDetection/valid/labels/pituitary_649_jpg.rf.7e31ac2655ed36becd74b70a2e8cc828.txt
================================================
3 0.525390625 0.3876953125 0.52734375 0.3642578125 0.5166015625 0.359375 0.4482421875 0.3671875 0.4453125 0.3935546875 0.42578125 0.4208984375 0.43359375 0.4443359375 0.4619140625 0.474609375 0.498046875 0.4775390625 0.5234375 0.4638671875 0.541015625 0.4306640625 0.541015625 0.4111328125 0.525390625 0.3876953125


================================================
FILE: TumorDetection/valid/labels/pituitary_657_jpg.rf.9a5199697d475febd4cfe5f0056a66dc.txt
================================================
3 0.474609375 0.4423828125 0.4404296875 0.416015625 0.3857421875 0.421875 0.36328125 0.4423828125 0.3515625 0.4677734375 0.349609375 0.5771484375 0.3759765625 0.595703125 0.396484375 0.5966796875 0.458984375 0.5712890625 0.462890625 0.5322265625 0.482421875 0.4892578125 0.474609375 0.4423828125


================================================
FILE: TumorDetection/valid/labels/pituitary_692_jpg.rf.9029e220dbb8f8bb70ce68e8a83c806f.txt
================================================
3 0.576171875 0.6025390625 0.58984375 0.5908203125 0.58984375 0.5771484375 0.5634765625 0.548828125 0.4853515625 0.556640625 0.4501953125 0.55078125 0.4306640625 0.5625 0.423828125 0.5771484375 0.421875 0.5927734375 0.435546875 0.6240234375 0.42578125 0.6552734375 0.4814453125 0.69921875 0.521484375 0.7080078125 0.5517578125 0.6953125 0.56640625 0.6787109375 0.58203125 0.6220703125 0.576171875 0.6025390625


================================================
FILE: TumorDetection/valid/labels/pituitary_698_jpg.rf.d435e3f15b867ebf52a22a63aaea5a5d.txt
================================================
3 0.591796875 0.5712890625 0.580078125 0.5283203125 0.5380859375 0.4921875 0.4599609375 0.490234375 0.42578125 0.5283203125 0.4140625 0.5595703125 0.4609375 0.6396484375 0.486328125 0.6552734375 0.5341796875 0.646484375 0.55078125 0.6298828125 0.560546875 0.6025390625 0.591796875 0.5712890625


================================================
FILE: TumorDetection/valid/labels/pituitary_700_jpg.rf.a321c78a245b0d7fa88bd88befa55ccb.txt
================================================
3 0.4736328125 0.390625 0.451171875 0.4130859375 0.451171875 0.4404296875 0.4794921875 0.474609375 0.50390625 0.4736328125 0.5234375 0.4462890625 0.5234375 0.4111328125 0.4990234375 0.388671875 0.4736328125 0.390625


================================================
FILE: TumorDetection/valid/labels/pituitary_705_jpg.rf.5937ed706720a93c4b3420ed66df3e54.txt
================================================
3 0.5107421875 0.376953125 0.4775390625 0.375 0.44921875 0.3935546875 0.4453125 0.4091796875 0.431640625 0.4208984375 0.43359375 0.4287109375 0.4560546875 0.4609375 0.4892578125 0.474609375 0.5078125 0.4736328125 0.5263671875 0.46875 0.53515625 0.4580078125 0.537109375 0.4111328125 0.5107421875 0.376953125


================================================
FILE: TumorDetection/valid/labels/pituitary_710_jpg.rf.289de300deb1a862a2afb4354da1f7fc.txt
================================================
3 0.5400390625 0.5703125 0.560546875 0.5400390625 0.564453125 0.5068359375 0.5458984375 0.46875 0.5263671875 0.462890625 0.5029296875 0.46875 0.4580078125 0.46484375 0.4375 0.4736328125 0.42578125 0.5029296875 0.427734375 0.5419921875 0.4443359375 0.55859375 0.4736328125 0.56640625 0.48046875 0.5810546875 0.4853515625 0.57421875 0.5126953125 0.578125 0.5166015625 0.5703125 0.5400390625 0.5703125
3 0.771484375 0.5966796875 0.791015625 0.5732421875 0.794921875 0.5263671875 0.80078125 0.5224609375 0.794921875 0.4873046875 0.783203125 0.4619140625 0.765625 0.4501953125 0.767578125 0.4169921875 0.744140625 0.3603515625 0.736328125 0.3564453125 0.736328125 0.3388671875 0.7197265625 0.3359375 0.7138671875 0.3203125 0.7060546875 0.326171875 0.6962890625 0.306640625 0.6904296875 0.310546875 0.6826171875 0.294921875 0.6748046875 0.302734375 0.67578125 0.2919921875 0.6494140625 0.275390625 0.619140625 0.2939453125 0.615234375 0.3232421875 0.595703125 0.3466796875 0.609375 0.3701171875 0.5888671875 0.376953125 0.58203125 0.3955078125 0.5859375 0.4521484375 0.59765625 0.4599609375 0.5908203125 0.45703125 0.587890625 0.4658203125 0.59765625 0.4658203125 0.6044921875 0.48828125 0.61328125 0.4736328125 0.615234375 0.4970703125 0.6328125 0.5029296875 0.6318359375 0.53125 0.669921875 0.5322265625 0.6611328125 0.54296875 0.6552734375 0.537109375 0.6328125 0.5400390625 0.6357421875 0.56640625 0.6591796875 0.560546875 0.68359375 0.5654296875 0.6767578125 0.572265625 0.6572265625 0.57421875 0.6513671875 0.568359375 0.642578125 0.5751953125 0.68359375 0.5810546875 0.66015625 0.5908203125 0.671875 0.5947265625 0.6611328125 0.609375 0.716796875 0.6123046875 0.7080078125 0.62109375 0.7041015625 0.61328125 0.6923828125 0.62109375 0.6845703125 0.61328125 0.6787109375 0.625 0.75390625 0.6259765625 0.771484375 0.6220703125 0.765625 0.6201171875 0.775390625 0.6025390625 0.771484375 0.5966796875


================================================
FILE: TumorDetection/valid/labels/pituitary_721_jpg.rf.9fd4133fe7ecaa1e04eafd2005a1acf2.txt
================================================
3 0.4501953125 0.4765625 0.41796875 0.5361328125 0.435546875 0.6025390625 0.4638671875 0.625 0.51953125 0.6259765625 0.552734375 0.5986328125 0.564453125 0.5068359375 0.5380859375 0.47265625 0.4501953125 0.4765625


================================================
FILE: TumorDetection/valid/labels/pituitary_733_jpg.rf.31501bcc82220ca2f87df8d1ccc85da2.txt
================================================
3 0.529296875 0.3857421875 0.5146484375 0.375 0.4892578125 0.373046875 0.453125 0.3994140625 0.45703125 0.4169921875 0.443359375 0.4892578125 0.4697265625 0.51953125 0.4990234375 0.517578125 0.517578125 0.5244140625 0.5546875 0.4951171875 0.556640625 0.4833984375 0.53515625 0.4501953125 0.529296875 0.3857421875


================================================
FILE: TumorDetection/valid/labels/pituitary_735_jpg.rf.8a8dd2f2ac0c494dc0aae8398042c87f.txt
================================================
3 0.580078125 0.4619140625 0.546875 0.4326171875 0.541015625 0.3857421875 0.5185546875 0.369140625 0.4970703125 0.369140625 0.4609375 0.3876953125 0.447265625 0.4130859375 0.451171875 0.4365234375 0.4296875 0.4521484375 0.439453125 0.4755859375 0.4404296875 0.505859375 0.4794921875 0.52734375 0.517578125 0.5341796875 0.5595703125 0.51953125 0.5712890625 0.5 0.578125 0.5048828125 0.580078125 0.4619140625


================================================
FILE: TumorDetection/valid/labels/pituitary_742_jpg.rf.d607bca9083caa3adad6e52ccd02d54c.txt
================================================
3 0.587890625 0.6083984375 0.5693359375 0.58203125 0.525390625 0.5537109375 0.5166015625 0.53515625 0.4794921875 0.529296875 0.466796875 0.5400390625 0.46484375 0.5654296875 0.431640625 0.5830078125 0.408203125 0.6455078125 0.4130859375 0.650390625 0.4169921875 0.640625 0.4306640625 0.646484375 0.4580078125 0.642578125 0.4775390625 0.673828125 0.515625 0.6826171875 0.5458984375 0.662109375 0.5810546875 0.66015625 0.595703125 0.6513671875 0.587890625 0.6083984375


================================================
FILE: TumorDetection/valid/labels/pituitary_746_jpg.rf.cdd8dbf8f485107bd5c72e7ac134ce13.txt
================================================
3 0.5263671875 0.4453125 0.4970703125 0.451171875 0.48046875 0.4677734375 0.47265625 0.4931640625 0.4453125 0.5068359375 0.4453125 0.5244140625 0.4638671875 0.5390625 0.5185546875 0.533203125 0.548828125 0.5419921875 0.5625 0.5244140625 0.5234375 0.4912109375 0.54296875 0.4697265625 0.5263671875 0.4453125


================================================
FILE: TumorDetection/valid/labels/pituitary_748_jpg.rf.747584aa30b5d6e4463549cf7e44fe74.txt
================================================
3 0.5263671875 0.447265625 0.4990234375 0.447265625 0.486328125 0.4599609375 0.482421875 0.4853515625 0.4453125 0.5009765625 0.447265625 0.5322265625 0.458984375 0.5380859375 0.4775390625 0.53125 0.5244140625 0.537109375 0.560546875 0.5263671875 0.5263671875 0.447265625


================================================
FILE: TumorDetection/valid/labels/pituitary_752_jpg.rf.68e1176f678b63c866968b1a53dd0a16.txt
================================================
3 0.58203125 0.4755859375 0.5732421875 0.466796875 0.5244140625 0.474609375 0.51953125 0.4365234375 0.4833984375 0.431640625 0.4609375 0.4482421875 0.478515625 0.4833984375 0.4580078125 0.48828125 0.4365234375 0.4765625 0.431640625 0.4873046875 0.44921875 0.5400390625 0.447265625 0.5595703125 0.501953125 0.5712890625 0.5439453125 0.552734375 0.5517578125 0.521484375 0.576171875 0.5205078125 0.58203125 0.4755859375


================================================
FILE: TumorDetection/valid/labels/pituitary_774_jpg.rf.edd6e4ca6dcef76261f4ed6f57264e2f.txt
================================================
3 0.4697265625 0.396484375 0.4375 0.4169921875 0.4375 0.4384765625 0.4501953125 0.462890625 0.51953125 0.4658203125 0.53515625 0.4248046875 0.5224609375 0.3984375 0.5029296875 0.392578125 0.4697265625 0.396484375


================================================
FILE: TumorDetection/valid/labels/pituitary_775_jpg.rf.136418ce384ae4b66c20095012be9481.txt
================================================
3 0.537109375 0.4287109375 0.53515625 0.4052734375 0.5107421875 0.384765625 0.4580078125 0.3828125 0.435546875 0.3994140625 0.4375 0.4814453125 0.4501953125 0.5 0.5 0.5029296875 0.5166015625 0.498046875 0.53125 0.4775390625 0.52734375 0.4560546875 0.537109375 0.4287109375


================================================
FILE: TumorDetection/valid/labels/pituitary_777_jpg.rf.e6602fb2d738e4c622b2a7fa4b569073.txt
================================================
3 0.515625 0.4853515625 0.5146484375 0.458984375 0.4541015625 0.462890625 0.447265625 0.4677734375 0.44921875 0.4970703125 0.4375 0.5146484375 0.4453125 0.5576171875 0.4716796875 0.578125 0.505859375 0.5791015625 0.5185546875 0.576171875 0.537109375 0.5556640625 0.5390625 0.5068359375 0.515625 0.4853515625


================================================
FILE: TumorDetection/valid/labels/pituitary_781_jpg.rf.5343e96691d37277402dafb1a48ead92.txt
================================================
3 0.2626953125 0.63671875 0.224609375 0.6591796875 0.22265625 0.6865234375 0.2646484375 0.7265625 0.28515625 0.7275390625 0.31640625 0.6865234375 0.31640625 0.6611328125 0.2978515625 0.638671875 0.2626953125 0.63671875


================================================
FILE: TumorDetection/valid/labels/pituitary_782_jpg.rf.7827aa6732810e05fe211ce5e688cb01.txt
================================================
3 0.7412109375 0.611328125 0.724609375 0.6279296875 0.71875 0.6591796875 0.7470703125 0.68359375 0.76953125 0.6826171875 0.8037109375 0.6640625 0.814453125 0.6376953125 0.7744140625 0.607421875 0.7412109375 0.611328125


================================================
FILE: TumorDetection/valid/labels/pituitary_785_jpg.rf.f221f95749e78883efbf98b858af826b.txt
================================================
3 0.54296875 0.4599609375 0.5126953125 0.435546875 0.4658203125 0.427734375 0.4453125 0.4521484375 0.44140625 0.5341796875 0.4658203125 0.552734375 0.4921875 0.5517578125 0.5390625 0.5380859375 0.548828125 0.5244140625 0.552734375 0.4814453125 0.54296875 0.4599609375


================================================
FILE: TumorDetection/valid/labels/pituitary_7_jpg.rf.b42b94a81cba4fc6ff7c6de77efcfe2a.txt
================================================
3 0.58203125 0.5927734375 0.5205078125 0.56640625 0.4853515625 0.568359375 0.4443359375 0.5546875 0.4130859375 0.55859375 0.365234375 0.6298828125 0.3671875 0.6435546875 0.3837890625 0.65234375 0.4384765625 0.658203125 0.4697265625 0.689453125 0.513671875 0.6962890625 0.5341796875 0.69140625 0.564453125 0.6396484375 0.59765625 0.6201171875 0.58203125 0.5927734375
3 0.5263671875 0.486328125 0.4736328125 0.48828125 0.431640625 0.5205078125 0.4296875 0.5322265625 0.4423828125 0.552734375 0.482421875 0.5595703125 0.5126953125 0.55859375 0.533203125 0.5458984375 0.5390625 0.5146484375 0.5263671875 0.486328125


================================================
FILE: TumorDetection/valid/labels/pituitary_805_jpg.rf.695fc45c8441c4e2f33286a3c0954637.txt
================================================
3 0.55078125 0.4775390625 0.5302734375 0.470703125 0.4873046875 0.474609375 0.48828125 0.4814453125 0.46484375 0.5009765625 0.4765625 0.5458984375 0.5126953125 0.560546875 0.55078125 0.5595703125 0.560546875 0.5185546875 0.55859375 0.5029296875 0.546875 0.4951171875 0.55078125 0.4775390625


================================================
FILE: TumorDetection/valid/labels/pituitary_818_jpg.rf.aef4102ca82d145dcf9797b039db2c82.txt
================================================
3 0.64453125 0.4638671875 0.630859375 0.4404296875 0.6064453125 0.421875 0.5517578125 0.421875 0.52734375 0.4423828125 0.521484375 0.4580078125 0.521484375 0.4931640625 0.541015625 0.5712890625 0.5751953125 0.59375 0.615234375 0.5986328125 0.6337890625 0.58984375 0.650390625 0.5634765625 0.654296875 0.5263671875 0.64453125 0.4638671875


================================================
FILE: TumorDetection/valid/labels/pituitary_828_jpg.rf.348b0e189d063776ac400fd9ea5fde9a.txt
================================================
3 0.6328125 0.5263671875 0.6044921875 0.505859375 0.5849609375 0.505859375 0.5546875 0.5244140625 0.5234375 0.6123046875 0.5322265625 0.63671875 0.5771484375 0.65625 0.591796875 0.6552734375 0.6318359375 0.640625 0.654296875 0.5927734375 0.65234375 0.5693359375 0.6328125 0.5263671875


================================================
FILE: TumorDetection/valid/labels/pituitary_878_jpg.rf.208922f4eef293b44c8ab463033157a6.txt
================================================
3 0.5693359375 0.447265625 0.5419921875 0.4453125 0.52734375 0.4736328125 0.529296875 0.5595703125 0.5556640625 0.57421875 0.591796875 0.5751953125 0.603515625 0.5712890625 0.59765625 0.5517578125 0.599609375 0.4775390625 0.5693359375 0.447265625


================================================
FILE: TumorDetection/valid/labels/pituitary_885_jpg.rf.f255cf825b37083846017c6d4d24b9bd.txt
================================================
3 0.626953125 0.4580078125 0.6044921875 0.419921875 0.5888671875 0.41015625 0.5634765625 0.408203125 0.525390625 0.4384765625 0.513671875 0.4892578125 0.5302734375 0.521484375 0.57421875 0.5322265625 0.6005859375 0.525390625 0.6171875 0.5068359375 0.626953125 0.4833984375 0.626953125 0.4580078125


================================================
FILE: TumorDetection/valid/labels/pituitary_889_jpg.rf.2c119b815694de488c965a9cdfa23261.txt
================================================
3 0.5166015625 0.494140625 0.490234375 0.5126953125 0.490234375 0.5576171875 0.5068359375 0.572265625 0.541015625 0.5771484375 0.564453125 0.5458984375 0.56640625 0.5146484375 0.5419921875 0.498046875 0.5166015625 0.494140625


================================================
FILE: TumorDetection/valid/labels/pituitary_910_jpg.rf.f12685eb37f15d08ce604a496f526b3d.txt
================================================
3 0.5322265625 0.48828125 0.4443359375 0.50390625 0.416015625 0.5361328125 0.408203125 0.5615234375 0.41015625 0.5751953125 0.439453125 0.6005859375 0.44921875 0.6279296875 0.4658203125 0.646484375 0.515625 0.6552734375 0.537109375 0.6435546875 0.5859375 0.5634765625 0.5703125 0.5224609375 0.5322265625 0.48828125


================================================
FILE: TumorDetection/valid/labels/pituitary_911_jpg.rf.4f4db29616e6d538754dc7b63191190e.txt
================================================
3 0.533203125 0.3916015625 0.5107421875 0.375 0.4697265625 0.373046875 0.474609375 0.4501953125 0.51953125 0.4775390625 0.53515625 0.4658203125 0.548828125 0.4384765625 0.548828125 0.4189453125 0.533203125 0.3916015625


================================================
FILE: TumorDetection/valid/labels/pituitary_912_jpg.rf.8ee8f5860ad33785a7c479732467305a.txt
================================================
3 0.556640625 0.6357421875 0.55078125 0.6552734375 0.5634765625 0.671875 0.5859375 0.6748046875 0.603515625 0.6611328125 0.60546875 0.6298828125 0.5732421875 0.623046875 0.556640625 0.6357421875


================================================
FILE: TumorDetection/valid/labels/pituitary_917_jpg.rf.592ab4188f0cec2e64fc2e75213cef11.txt
================================================
3 0.5478515625 0.365234375 0.5224609375 0.35546875 0.4970703125 0.357421875 0.4775390625 0.369140625 0.4609375 0.3935546875 0.45703125 0.4482421875 0.4677734375 0.46484375 0.4912109375 0.478515625 0.5234375 0.4794921875 0.5517578125 0.46875 0.57421875 0.4482421875 0.580078125 0.4072265625 0.5478515625 0.365234375


================================================
FILE: TumorDetection/valid/labels/pituitary_933_jpg.rf.365b22a61ed488ab4dc23230d772900e.txt
================================================
3 0.54296875 0.3759765625 0.4599609375 0.369140625 0.44921875 0.4287109375 0.4609375 0.4443359375 0.5263671875 0.431640625 0.5380859375 0.3984375 0.54296875 0.4169921875 0.54296875 0.3759765625


================================================
FILE: TumorDetection/valid/labels/pituitary_937_jpg.rf.e4d8a601c65d84971df3794dd86bba7e.txt
================================================
3 0.4912109375 0.341796875 0.4716796875 0.349609375 0.455078125 0.3701171875 0.44921875 0.4013671875 0.4716796875 0.43359375 0.509765625 0.4501953125 0.544921875 0.4189453125 0.55078125 0.3798828125 0.5322265625 0.353515625 0.4912109375 0.341796875


================================================
FILE: TumorDetection/valid/labels/pituitary_990_jpg.rf.33995d119eb8785c89eaef7e652c2ca2.txt
================================================
3 0.4931640625 0.37109375 0.484375 0.3916015625 0.490234375 0.4169921875 0.5185546875 0.435546875 0.548828125 0.4384765625 0.572265625 0.4208984375 0.568359375 0.3876953125 0.5146484375 0.365234375 0.4931640625 0.37109375

