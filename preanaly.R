# omit all the rows contains 0
dat.df[dat.df==0] <- NA
dat.df <- na.omit(dat.df)

# the average npp changing of the whole zone
mean <- c()
for (i in seq(3,23,1)){
  me <- mean(dat.df[,i])
  mean <- append(mean,me)
}
plot(mean,type='l')

# the average npp changing of the whole zone of each type of tree
tree <- c()
for (i in seq(3,23,1)){
  tr <- tapply(dat.df[,i],dat.df[,1],mean)
  tree <- rbind(tree,tr) 
}
write.table(tree,'tree.csv')

avgtr <- c()
for (i in seq(1,32)){
  avgtr <- append(avgtr,mean(tree[,i]))
}
  
