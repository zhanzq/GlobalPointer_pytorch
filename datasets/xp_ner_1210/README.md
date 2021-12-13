### 2021-12-10标注数据更新

**主要更新项目**

1. 将**第?个**未标注的数据标注为**poiIndex**

**示例1：**

```{"text": "第一个", "label": []} ==> {"text": "第一个", "label": [["poiIndex", "第一个"]]}```

**更新详情：**

```json
# case 1, line 17:
 original: {"text": "第一家吧排下号小P", "label": []}
 modified: {"text": "第一家吧排下号小P", "label": [["poiIndex", "第一家"]]}
# case 2, line 441:
 original: {"text": "选择第一家吧", "label": [["poiName", "第一家"]]}
 modified: {"text": "选择第一家吧", "label": [["poiIndex", "第一家"]]}
# case 3, line 484:
 original: {"text": "嗯第二个吧", "label": []}
 modified: {"text": "嗯第二个吧", "label": [["poiIndex", "第二个"]]}
# case 4, line 509:
 original: {"text": "第一个人均价格在100元以内吗", "label": []}
 modified: {"text": "第一个人均价格在100元以内吗", "label": [["poiIndex", "第一个"]]}
# case 5, line 1637:
 original: {"text": "要不我们就选第一个吧", "label": []}
 modified: {"text": "要不我们就选第一个吧", "label": [["poiIndex", "第一个"]]}
# case 6, line 1789:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 7, line 2707:
 original: {"text": "请问第一个餐厅排号人数是多少", "label": []}
 modified: {"text": "请问第一个餐厅排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 8, line 2835:
 original: {"text": "那选择第二条开始导航", "label": []}
 modified: {"text": "那选择第二条开始导航", "label": [["poiIndex", "第二条"]]}
# case 9, line 3132:
 original: {"text": "第一个现在要排多久", "label": []}
 modified: {"text": "第一个现在要排多久", "label": [["poiIndex", "第一个"]]}
# case 10, line 3474:
 original: {"text": "第一家是吃什么的看评分不错看起来也好好吃的样子", "label": [["poiName", "第一家"]]}
 modified: {"text": "第一家是吃什么的看评分不错看起来也好好吃的样子", "label": [["poiIndex", "第一家"]]}
# case 11, line 3671:
 original: {"text": "请问第四个几点开店", "label": []}
 modified: {"text": "请问第四个几点开店", "label": [["poiIndex", "第四个"]]}
# case 12, line 4574:
 original: {"text": "请问第一个餐厅排号人数是多少", "label": []}
 modified: {"text": "请问第一个餐厅排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 13, line 4599:
 original: {"text": "那第三家吧", "label": [["poiName", "第三家"]]}
 modified: {"text": "那第三家吧", "label": [["poiIndex", "第三家"]]}
# case 14, line 5570:
 original: {"text": "我看第四家就挺好的呀再看看吧", "label": [["poiName", "第四家"]]}
 modified: {"text": "我看第四家就挺好的呀再看看吧", "label": [["poiIndex", "第四家"]]}
# case 15, line 6077:
 original: {"text": "去第二个吧", "label": []}
 modified: {"text": "去第二个吧", "label": [["poiIndex", "第二个"]]}
# case 16, line 6219:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 17, line 6334:
 original: {"text": "那就看看第四家吧", "label": [["poiName", "第四家"]]}
 modified: {"text": "那就看看第四家吧", "label": [["poiIndex", "第四家"]]}
# case 18, line 6883:
 original: {"text": "我看看第一个", "label": []}
 modified: {"text": "我看看第一个", "label": [["poiIndex", "第一个"]]}
# case 19, line 7398:
 original: {"text": "第二条", "label": []}
 modified: {"text": "第二条", "label": [["poiIndex", "第二条"]]}
# case 20, line 7441:
 original: {"text": "第二个要等多久", "label": []}
 modified: {"text": "第二个要等多久", "label": [["poiIndex", "第二个"]]}
# case 21, line 8443:
 original: {"text": "去第一个要多远", "label": []}
 modified: {"text": "去第一个要多远", "label": [["poiIndex", "第一个"]]}
# case 22, line 8539:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 23, line 8802:
 original: {"text": "第一个排号人数怎样", "label": []}
 modified: {"text": "第一个排号人数怎样", "label": [["poiIndex", "第一个"]]}
# case 24, line 8974:
 original: {"text": "第二个", "label": []}
 modified: {"text": "第二个", "label": [["poiIndex", "第二个"]]}
# case 25, line 9462:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 26, line 9565:
 original: {"text": "第一个有多少人在排号", "label": []}
 modified: {"text": "第一个有多少人在排号", "label": [["poiIndex", "第一个"]]}
# case 27, line 9913:
 original: {"text": "别了吧就第三家吧吃个饭而已又不是挑女婿磨磨唧唧的", "label": [["poiName", "第三家"]]}
 modified: {"text": "别了吧就第三家吧吃个饭而已又不是挑女婿磨磨唧唧的", "label": [["poiIndex", "第三家"]]}
# case 28, line 10671:
 original: {"text": "第一个服务区还有没有停车位", "label": [["location", "服务区"]]}
 modified: {"text": "第一个服务区还有没有停车位", "label": [["poiIndex", "第一个"], ["location", "服务区"]]}
# case 29, line 11458:
 original: {"text": "导航去第一个小P", "label": []}
 modified: {"text": "导航去第一个小P", "label": [["poiIndex", "第一个"]]}
# case 30, line 11546:
 original: {"text": "第一个店现在排号人数是多少", "label": []}
 modified: {"text": "第一个店现在排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 31, line 11609:
 original: {"text": "就第一家吧这家店在营业", "label": [["poiName", "第一家"]]}
 modified: {"text": "就第一家吧这家店在营业", "label": [["poiIndex", "第一家"]]}
# case 32, line 11626:
 original: {"text": "第一条", "label": []}
 modified: {"text": "第一条", "label": [["poiIndex", "第一条"]]}
# case 33, line 13410:
 original: {"text": "导航第二个", "label": []}
 modified: {"text": "导航第二个", "label": [["poiIndex", "第二个"]]}
# case 34, line 13424:
 original: {"text": "第一个人均价格是多少钱", "label": []}
 modified: {"text": "第一个人均价格是多少钱", "label": [["poiIndex", "第一个"]]}
# case 35, line 13895:
 original: {"text": "第四个的均价在300元内吗", "label": []}
 modified: {"text": "第四个的均价在300元内吗", "label": [["poiIndex", "第四个"]]}
# case 36, line 14014:
 original: {"text": "那第二家呢", "label": [["poiName", "第二家"]]}
 modified: {"text": "那第二家呢", "label": [["poiIndex", "第二家"]]}
# case 37, line 14097:
 original: {"text": "第一条", "label": []}
 modified: {"text": "第一条", "label": [["poiIndex", "第一条"]]}
# case 38, line 14623:
 original: {"text": "第一个点都德要排多久", "label": [["poiName", "点都德"]]}
 modified: {"text": "第一个点都德要排多久", "label": [["poiIndex", "第一个"], ["poiName", "点都德"]]}
# case 39, line 14994:
 original: {"text": "第一个有停车场吗", "label": []}
 modified: {"text": "第一个有停车场吗", "label": [["poiIndex", "第一个"]]}
# case 40, line 15100:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 41, line 15319:
 original: {"text": "不用了就第三家吧远点就远点吧", "label": [["poiName", "第三家"]]}
 modified: {"text": "不用了就第三家吧远点就远点吧", "label": [["poiIndex", "第三家"]]}
# case 42, line 15445:
 original: {"text": "去第二家吃烧烤吧看着好好吃的样子", "label": [["poiName", "第二家"], ["type", "烧烤"]]}
 modified: {"text": "去第二家吃烧烤吧看着好好吃的样子", "label": [["poiIndex", "第二家"], ["type", "烧烤"]]}
# case 43, line 15566:
 original: {"text": "第一个店现在排号人数是多少", "label": []}
 modified: {"text": "第一个店现在排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 44, line 15977:
 original: {"text": "第一个人均价格在100元内吗", "label": []}
 modified: {"text": "第一个人均价格在100元内吗", "label": [["poiIndex", "第一个"]]}
# case 45, line 16004:
 original: {"text": "请问第一个餐厅排号人数是多少", "label": []}
 modified: {"text": "请问第一个餐厅排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 46, line 16108:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 47, line 16163:
 original: {"text": "第二家呢我觉得不错", "label": [["poiName", "第二家"]]}
 modified: {"text": "第二家呢我觉得不错", "label": [["poiIndex", "第二家"]]}
# case 48, line 16326:
 original: {"text": "第一家评分超高值得去啊", "label": [["poiName", "第一家"]]}
 modified: {"text": "第一家评分超高值得去啊", "label": [["poiIndex", "第一家"]]}
# case 49, line 16822:
 original: {"text": "第一个好像不错挺好吃的东西看起来", "label": []}
 modified: {"text": "第一个好像不错挺好吃的东西看起来", "label": [["poiIndex", "第一个"]]}
# case 50, line 16947:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 51, line 18405:
 original: {"text": "第一个人均价格是多少钱", "label": []}
 modified: {"text": "第一个人均价格是多少钱", "label": [["poiIndex", "第一个"]]}
# case 52, line 18534:
 original: {"text": "那排号第二个吧", "label": []}
 modified: {"text": "那排号第二个吧", "label": [["poiIndex", "第二个"]]}
# case 53, line 19171:
 original: {"text": "第三家都好就是远了一点就不去了", "label": [["poiName", "第三家"]]}
 modified: {"text": "第三家都好就是远了一点就不去了", "label": [["poiIndex", "第三家"]]}
# case 54, line 19855:
 original: {"text": "那第二家呢我看着也挺不错的", "label": [["poiName", "第二家"]]}
 modified: {"text": "那第二家呢我看着也挺不错的", "label": [["poiIndex", "第二家"]]}
# case 55, line 20509:
 original: {"text": "第一个多人吗帮我查查", "label": []}
 modified: {"text": "第一个多人吗帮我查查", "label": [["poiIndex", "第一个"]]}
# case 56, line 20530:
 original: {"text": "第一个需要排队吗", "label": []}
 modified: {"text": "第一个需要排队吗", "label": [["poiIndex", "第一个"]]}
# case 57, line 20572:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 58, line 21189:
 original: {"text": "那就去第三家吧", "label": [["poiName", "第三家"]]}
 modified: {"text": "那就去第三家吧", "label": [["poiIndex", "第三家"]]}
# case 59, line 21244:
 original: {"text": "是的珍姐龙虾第三家分店开在员村那里了我们去尝尝", "label": [["poiName", "珍姐龙虾第三家分店"], ["location", "员村"]]}
 modified: {"text": "是的珍姐龙虾第三家分店开在员村那里了我们去尝尝", "label": [["poiName", "珍姐龙虾第三家分店"], ["location", "员村"]]}
# case 60, line 21330:
 original: {"text": "那就订第二个", "label": []}
 modified: {"text": "那就订第二个", "label": [["poiIndex", "第二个"]]}
# case 61, line 22374:
 original: {"text": "请问第一个餐厅排号人数是多少", "label": []}
 modified: {"text": "请问第一个餐厅排号人数是多少", "label": [["poiIndex", "第一个"]]}
# case 62, line 22468:
 original: {"text": "我们去第一家吃吧", "label": [["poiName", "第一家"]]}
 modified: {"text": "我们去第一家吃吧", "label": [["poiIndex", "第一家"]]}
# case 63, line 22522:
 original: {"text": "第一家店是吃什么的啊", "label": [["poiName", "第一家"]]}
 modified: {"text": "第一家店是吃什么的啊", "label": [["poiIndex", "第一家"]]}
# case 64, line 22566:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 65, line 22665:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 66, line 22677:
 original: {"text": "去第一个", "label": []}
 modified: {"text": "去第一个", "label": [["poiIndex", "第一个"]]}
# case 67, line 23138:
 original: {"text": "那好吧就去第三家吧", "label": [["poiName", "第三家"]]}
 modified: {"text": "那好吧就去第三家吧", "label": [["poiIndex", "第三家"]]}
# case 68, line 23622:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 69, line 24929:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 70, line 24939:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 71, line 25150:
 original: {"text": "第一个", "label": []}
 modified: {"text": "第一个", "label": [["poiIndex", "第一个"]]}
# case 72, line 25237:
 original: {"text": "选第二个吧比较近", "label": []}
 modified: {"text": "选第二个吧比较近", "label": [["poiIndex", "第二个"]]}
# case 73, line 25922:
 original: {"text": "行就吃那个天下第一家吧", "label": [["poiName", "天下第一家"]]}
 modified: {"text": "行就吃那个天下第一家吧", "label": [["poiName", "天下第一家"]]}
# case 74, line 25923:
 original: {"text": "行就吃那个倒数第四家吧", "label": [["poiName", "倒数第四家"]]}
 modified: {"text": "行就吃那个倒数第四家吧", "label": [["poiIndex", "倒数第四家"]]}
# case 75, line 25924:
 original: {"text": "行就吃那个倒数第四家吧", "label": [["poiName", "倒数第四家"]]}
 modified: {"text": "行就吃那个倒数第四家吧", "label": [["poiIndex", "倒数第四家"]]}
# case 76, line 25955:
 original: {"text": "我买了第一家的代金券我们去那吃吧", "label": []}
 modified: {"text": "我买了第一家的代金券我们去那吃吧", "label": [["poiIndex", "第一家"]]}
# case 77, line 25968:
 original: {"text": "第一家的臭豆腐看着流口水了", "label": [["dishName", "臭豆腐"]]}
 modified: {"text": "第一家的臭豆腐看着流口水了", "label": [["poiIndex", "第一家"], ["dishName", "臭豆腐"]]}
# case 78, line 25969:
 original: {"text": "第一家的臭豆腐看着流口水了", "label": [["dishName", "臭豆腐"]]}
 modified: {"text": "第一家的臭豆腐看着流口水了", "label": [["poiIndex", "第一家"], ["dishName", "臭豆腐"]]}
# case 79, line 25970:
 original: {"text": "第一家的臭豆腐看着流口水了", "label": [["dishName", "臭豆腐"]]}
 modified: {"text": "第一家的臭豆腐看着流口水了", "label": [["poiIndex", "第一家"], ["dishName", "臭豆腐"]]}
```
