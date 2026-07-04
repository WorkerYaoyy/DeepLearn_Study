语句:

```powershell
git init
# 替换为你的用户名
git config --global user.name "HYH"
# 替换为你的平台注册邮箱
git config --global user.email "2533957897@qq.com"
git add .
git commit -m "提交备注" #创建版本
git restore . #回退
git branch #查看当前分支
git switch -c "分支名" #创建分支
git switch name #切换回分支
git push . 
```

注意:

git add之后不能直接push必须git commit后由工作区进入本地仓库后才可上传至GitHub，否则会出现”Everything up-to-date“没东西可以更新。
