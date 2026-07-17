Chinese character involved in this project, mind encoding!
使用myml python环境,解释器位于C:\Users\zyx20\anaconda3\envs\myml\python.exe
我需要你帮助我开发CWGAN-GP的翼型设计神经网络。神经网络中包含贝塞尔曲线层
当前终端为powershell 5.1, 不支持 && 以及||操作符

在开始较复杂的非只读任务前，向用户询问详细的技术细节。完成和用户的alignment后寻求用户同意，并在docs文件夹中写入/修改对应模块的.md定义文件。保持定义简洁

开发规则：
1. DRY 原则 (Don't Repeat Yourself)
拒绝知识的重复。系统中的每一个功能点、算法或配置，都应有且仅有一个权威定义。禁止在多个地方手动同步相同的逻辑.

平衡点： 避免为了 DRY 而引入过度复杂的泛型或多层继承。如果消除重复会导致代码可读性急剧下降，请优先选择代码的清晰度，并辅助以显式注释.

2. 单一来源(Single Source of Truth)
常量、魔术字符串, 数据库 Schema 必须定义在集中配置文件中

3. Fail-Fast 机制暴露错误
禁止过度防御性编程，不使用 config.get('max_workers', 4)的默认参数，必须让潜在的错误直接通过报错暴露出来


mistakes:
1. 当在CLI执行 Python 代码块时，必须遵循以下原则：不要在 f-string 中使用引号嵌套；禁止在 {} 内出现反斜杠；严格引号分层。禁止在 f-string 的大括号内使用任何引号或反斜杠。如果需要打印字典内容，使用 print('text', dict['key']) 这种多参数形式，不要在字符串内部嵌套转义引号

2. 不要尝试执行单行复杂命令，换行语法容易出错。2行以上命令必须写出临时代码再执行
