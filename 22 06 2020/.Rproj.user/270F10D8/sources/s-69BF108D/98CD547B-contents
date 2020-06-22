#
#
#         Example of comparing 2 groups
#
#

# ---------------------- set environment ---------------------------------
WD           = "C:/work/"
PATH         = WD
project.dir  = "demo/"

path.dir.in       = paste0(WD, project.dir, "data_in/")
path.dir.out      = paste0(WD, project.dir, "data_res/")
path.pics.out     = paste0(WD, project.dir, "pics_res/")

# include funcs
path         = paste0(WD, project.dir, 'cmp_function.R')
source(path)
# ---------------------- set environment ---------------------------------



cmp.groups = list( 'CYP2C19_2'     = list(col = 'CYP2C19_2'),
                   'CYP2C19_2_LOF' = list(col = '2+3 vs. 1: CYP2C19_2, 1 == 0, 2+3 == 1'),
                   
                   'CYP2C19_3'     = list(col = 'CYP2C19_3'),
                   'CYP2C19_3_LOF' = list(col = '2+3 vs. 1: CYP2C19_3, 1 == 0, 2+3 == 1'),
                   'CYP2C19_17'     = list(col = 'CYP2C19_17'),
                   'CYP2C19_17_LOF' = list(col = '2+3 vs. 1: CYP2C19_17, 1 == 0, 2+3 == 1'),
                   'CYP2C19_2_3_norm_vs_LOF' = list(col = '@ 2+3 vs. 1: CYP2C19_2 OR CYP2C19_3'),
                   'CYP2C19_2_17_norm_vs_LOF' = list(col = '@ 2+3 vs. 1: CYP2C19_2 OR CYP2C19_17'),
                   'CYP2C19_3_17_norm_vs_LOF' = list(col = '@ 2+3 vs. 1: CYP2C19_3 OR CYP2C19_17'),
                   'CYP2C19_2_3_norm_vs_gomo_LOF' = list(col = '@ 3 vs. 1: CYP2C19_2 OR CYP2C19_3'),
                   'CYP2C19_2_17_norm_vs_gomo_LOF' = list(col = '@ 3 vs. 1: CYP2C19_2 OR CYP2C19_17'),
                   'CYP2C19_3_17_norm_vs_gomo_LOF' = list(col = '@ 3 vs. 1: CYP2C19_3 OR CYP2C19_17')
) 
#
#  ----- Step 1. Create list of comparing groups -----------------------
#

#
#  ----- Step 2. Compare complications in groups -----------------------
#

ls = list(list(id = 33, type = 'bin'))
ls = append.descr(ls, ids  = c(34,36,38,39,42,44,79,81,85,86), type = 'bin')
ls = append.descr(ls, ids  = c(49,51,53,55,80,82), type = 'bin')

#
#  ----- Tukey test -----
#
res.table = NULL

for( groups in names(cmp.groups)){
  data.temp    = data.Proc
  id.col       = cmp.groups[[groups]][['col']]
  data.temp[[id.col]] = as.character(data.temp[[id.col]])
  t.points     = as.character(sort(unique(data.temp[[id.col]])))
  len.t.points = length(t.points)
  
  for( i in 1:(len.t.points-1)){
    for(j in (i+1):len.t.points){
      cat("\n")
      cat("dimension:")
      cat(dim(data.temp))
      cat("\n")
      
      group.No.1 = t.points[i]
      group.No.2 = t.points[j]
      cat( group)
      cat(": ")
      cat(group.No.1)
      cat(" ")
      cat(group.No.2)
      cat("\n")
      
      #group.No.1 = as.character(unique(data.temp$t.point)[i])
      #group.No.2 = as.character(unique(data.temp$t.point)[j])
      
      F.NAME = paste0( 
        as.character(groups),
        "__",
        gsub('[\\. ]','_', group.No.1),
        "_",
        gsub('[\\. ]','_', group.No.2)
      )
      
      
      #debug(cmp.2group.all.vars) 
      res.cmp = cmp.2group.all.vars(id.group     = id.col, 
                                    group.labels = c(group.No.1, group.No.2, group.No.3),
                                    columns.list = ls,
                                    data         = data.temp, 
                                    PATH         = "C:/work/demo/data_res", 
                                    FILENAME     = "m", 
                                    #paired       = F, 
                                    result = NULL
      )
      res.cmp = cbind(names = rownames(res.cmp),res.cmp)
      
      
      res.cmp = res.cmp %>% 
        select(1,2,3,4,5,6,7) %>%
        rename_at(c(1,2,3,4,5,6,7),
                  ~c('names',
                     paste0('n (%): ',group.No.1),
                     paste0('values: ',group.No.1),
                     paste0('n (%): ',group.No.2),
                     paste0('values: ',group.No.2),
                     paste0('difference: ', group.No.1, '-',group.No.2),
                     paste0('p-value: ',    group.No.1, '-', group.No.2)
                  )
        )
      
      if( (i == 1) & (j == 2)){
        res.table = res.cmp
      } else{
        res.table = left_join(res.table, res.cmp,by='names')
      }
    }
  }
  
  p.table = res.table %>%
    select(starts_with('p-value: '))
  
  p.common = NULL
  for (ii in 1:dim(p.table)[2] )
  {
    xx = paste0(gsub('p-value: ','',colnames(p.table)[ii]),
                ': ',
                p.table[[ii]],
                '\n'
    )
    if(ii == 1){
      p.common = xx
    }else{
      p.common =  paste0(p.common, xx)
    }
  }
  
  diff.table = res.table %>%
    select(starts_with('difference: '))
  diff.common = NULL
  for (ii in 1:dim(diff.table)[2] )
  {
    xx = paste0(gsub('difference: ','',colnames(diff.table)[ii]),
                ': ',
                diff.table[[ii]],
                '\n'
    )
    if(ii == 1){
      diff.common = xx
    }else{
      diff.common =  paste0(diff.common, xx)
    }
  }
  
  res.table = cbind(res.table, diff.common, p.common)
  F.NAME = paste0(  path.dir.out,
                    'COMMON_TABLE_',
                    gsub('\\. ', '_', as.character(groups)),
                    '.csv'
  )
  cat(F.NAME)
  res.table %>% write.csv2(file=F.NAME)
  
}
#
#  ----- COMPARE MANY GROUPS IN ONE TIME POINT   -----------------------



