语句:

```powershell
git init
git clone https://XXXX  #连接上Github上的远程仓库
git remote add origin https:// #设置远程仓库地址
git config --global core.autocrlf true #屏蔽换行警告
# 替换为你的用户名
git config --global user.name "HYH"
# 替换为你的平台注册邮箱
git config --global user.email "2533957897@qq.com"
git add .
git commit -m "提交备注" #创建版本
git restore . #回退
git log #查看提交记录
git branch #查看当前分支
git switch -c "分支名" #创建分支
git switch name #切换回分支
git push . 
git push #.代表本地分支 无则代表远程分支
git pull origin main #拉取远程仓库最新代码
git remote -v #查看连接的远程仓库
```

从零开始的使用流程

如下在选定文件夹右键打开powershell运行以下代码

```powershell
git init

git config --global user.name #替换为你的用户名
git config --global user.email #替换为你的平台注册邮箱

git commit -m "first commit"
git branch -M main #将当前分支名改为main
git remote add origin https://XXXX #绑定远程仓库
git push -u origin main #提交到github

```

注意:

git add之后不能直接push必须git commit后由工作区进入本地仓库后才可上传至GitHub，否则会出现”Everything up-to-date“没东西可以更新。

在修改GitHub账户时会出现红色提示，只要账户正确就不用管。
