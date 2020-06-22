se <- function(x) 
{
    sd(x, na.rm = TRUE)/sqrt(length(x))
}

median.my <- function (vec, na.rm.my = TRUE)
{
    if(is.factor(vec)) return (NA)
    return (median(vec, na.rm=TRUE))
}

quantile.my <- function(vec, probs = 0.25, na.rm = TRUE)
{
    if(is.factor(vec)) return (NA)
    return (quantile(vec, probs, na.rm = TRUE ))
    
}

quantile.interval.my <- function(vec, 
                                 prob.min    = 0.25, 
                                 prob.max    = 0.75, 
                                 na.rm       = TRUE, 
                                 pos         = 2
)
{
    require(plyr)
    require(magrittr)
    
    qu.min = vec %>% quantile(prob.min, na.rm = TRUE) %>% round(digits = pos)
    qu.max = vec %>% quantile(prob.max, na.rm = TRUE) %>% round(digits = pos)
    
    return (paste(c( '[', as.character(qu.min),'; ',as.character(qu.max), ']'), collapse=''))
}

min.my <- function(vec, na.rm = TRUE)
{
    if(is.factor(vec)) return (NA)
    return (min(vec, na.rm=TRUE))
    
}

max.my <- function(vec, na.rm = TRUE)
{
    if(is.factor(vec)) return (NA)
    return (max(vec, na.rm=TRUE))
    
}

bin.pos.my <- function(vec)
{
    return(sum(vec == 1, na.rm=TRUE))
}

bin.neg.my <- function(vec)
{
    return(length(vec)-bin.pos.my(vec))
    
}

bin.med.my <- function(vec)
{
    require("binom")
    res <- binom.confint(bin.pos.my(vec),length(vec),methods = "wilson")
    return((100.0*as.numeric(res$mean)) %>% round(digits = 2))
    
}

bin.ci.my <- function(vec, pos = 2, type.brackets = "()")
{
    res <- binom.confint(bin.pos.my(vec), length(vec),methods = "wilson")
    left   = (100.0*as.numeric(res$lower)) %>% round(digits = pos)
    right  = (100.0*as.numeric(res$upper)) %>% round(digits = pos)
    
    if (type.brackets == "()")
    {
        return (paste(c( '(', as.character(left),'; ',as.character(right), ')'), collapse=''))
    }
    
    if (type.brackets == "[]")
    {
        return (paste(c( '[', as.character(left),'; ',as.character(right), ']'), collapse=''))
    }
    if (type.brackets == "[%]")
    {
        return (paste(c( '[', as.character(left),'%; ',as.character(right), '%]'), collapse=''))
    }
}


bin.ci.my.num <- function(vec, b = "l")
{
    res <- binom.confint(bin.pos.my(vec), length(vec),methods = "wilson")
    left   = (as.numeric(res$lower)) %>% round(digits = 2)
    right  = (as.numeric(res$upper)) %>% round(digits = 2)
    
    if(b=="l"){
        return (left)
    }else{
        return (right)   
    }
    
}

bin.ci.my.l<- function(vec)
{
    return (bin.ci.my.num(vec, b="l"))
}
bin.ci.my.r<- function(vec)
{
    return (bin.ci.my.num(vec, b="r"))
}

bin.my = function(vec, digits=2)
{
    if ( all(is.na(vec)) == TRUE ) return("NA")
    res = paste0( round(bin.pos.my(vec),digits),", ",
                  as.character(round(bin.med.my(vec),digits)), '% ', # %,
                  bin.ci.my(vec, pos = digits, type.brackets = "[%]") 
    )  
    
}


#' Title
#'
#' @param id.group     id columns in data frame whith groups labels
#' @param group.labels vector of group labels
#' @param id.columns   vector containing id's of comparing columns 
#' @param type.columns vector containing types: "num","bin" or "cat" of comparing columns 
#' @param data         data frame 
#' @param PATH         path to write result
#' @param FILENAME     filename for result
#' @param dbg          0 - off, 1- onn debug mode 
#'
#' @return
#' @export
#'
#' @examples
cmp.2group.all = function(id.group, 
                          group.labels = NULL,
                          id.columns,
                          type.columns,
                          data, 
                          PATH = NULL, 
                          FILENAME = NULL, 
                          result = NULL,
                          dbg    = 1
)
{
    require(dplyr)
    
    #dt1 = data[data[[id.group]] == group.labels[1], ]
    #dt2 = data[data[[id.group]] == group.labels[2], ]
    
    result = data.frame( n.1         = character(),
                         group.1     = character(),
                         n.2         = character(),
                         group.2     = character(),
                         n.3         = character(),
                         group.3     = character(),
                         diff        = character(),
                         p.value     = character(),
                         method      = character()
    )
    colnames(result)[1] = group.labels[1]
    colnames(result)[2] = group.labels[2]
    colnames(result)[3] = group.labels[3]
    
    for( i in 1:length(id.columns))
    {
        
        i.col  = id.columns[i]
        if (dbg == 1){
            cat("\nCurrent column | ", colnames(data)[i.col],"| id = ", i.col)
        }
        
        dt     = parse.data.frame.with.na( dt = data,
                                           groups.column = id.group,
                                           group.labels  = group.labels,
                                           values.column = i.col
        )
        
        # 1 - id.group
        # 2 - i.col
        x = dt[dt[[1]] == group.labels[1], ][,2]
        y = dt[dt[[1]] == group.labels[2], ][,2]
        z = dt[dt[[1]] == group.labels[3], ][,2]
        n.x = na.count.string(x)
        n.y = na.count.string(y)
        n.z = na.count.string(z)
        
        dt     = parse.data.frame(  dt            = data,
                                    groups.column = id.group,
                                    group.labels  = group.labels,
                                    values.column = i.col
        )
        # 1 - id.group
        # 2 - i.col
        x = dt[dt[[1]] == group.labels[1], ][,2]
        y = dt[dt[[1]] == group.labels[2], ][,2]
        z = dt[dt[[1]] == group.labels[3], ][,2]
        
        if(dbg == 1){
            
            if(length(x) < 3 ) {
                cat ("WARNING in group: ", group.labels[1], "column:  ", colnames(data)[i.col], "\n" )
                cat ("first group too small")
                #stop("first group too small")
                x = NA
            }
            if(length(y) < 3 ) {
                cat ("WARNING in group: ", group.labels[2], "column:  ", colnames(data)[i.col],  "\n" )
                cat ("second group too small" )
                y = NA
                #stop("first group too small")
            }
            if(length(z) < 3 ) {
                cat ("WARNING in group: ", group.labels[3], "column:  ", colnames(data)[i.col],  "\n" )
                cat ("third group too small" )
                z = NA
                #stop("first group too small")
            }
        }
        
        if(type.columns[i] == "num")
        {
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE) &
                (all(is.na(z)) == FALSE)
            )
            {    
                #apply U-test Mana-Uitni
                res         = wilcox.test(x, y, paired=FALSE, conf.int = TRUE)
                res.stat    = round(res$statistic, 4)
                res.p.value = p_value_formatted(res$p.value)
                
                res.mu      = -round(res$estimate,  2)
                res.ci      = paste(c( '[', 
                                       as.character(-round(res$conf.int[2],2)),
                                       '; ',
                                       as.character(-round(res$conf.int[1],2)), 
                                       ']'), collapse=''
                )
                res.df = data.frame( n.1      = n.x,
                                     group.1  = paste0(as.character(median.my(x)),
                                                       " ",
                                                       quantile.interval.my(x),
                                                       " ",
                                                       as.character(round(mean(x, na.rm = TRUE), 2)),
                                                       "?",
                                                       as.character(round(sd(x, na.rm = TRUE), 2))
                                     ),
                                     n.2      = n.y,
                                     group.2  = paste0(as.character(median.my(y)),
                                                       " ",
                                                       quantile.interval.my(y),
                                                       " ",
                                                       as.character(round(mean(y, na.rm = TRUE), 2)),
                                                       "?",
                                                       as.character(round(sd(y, na.rm = TRUE), 2))
                                     ),
                                     group.3  = paste0(as.character(median.my(z)),
                                                       " ",
                                                       quantile.interval.my(z),
                                                       " ",
                                                       as.character(round(mean(z, na.rm = TRUE), 2)),
                                                       "?",
                                                       as.character(round(sd(z, na.rm = TRUE), 2))
                                     ),
                                     diff     = paste0( as.character(res.mu)," ", res.ci),
                                     p.value     = as.character(res.p.value),
                                     method      = "U-test Mann-Whitney"
                )
            }
            if(
                (all(is.na(x)) == TRUE) & 
                (all(is.na(y)) == FALSE) &
                (all(is.na(z)) == FALSE)
            )
            {
                res.df = data.frame( n.1      = n.x,
                                     group.1  = "NA",
                                     n.2      = n.y,
                                     group.2  = paste0( as.character(median.my(y)),
                                                        " ",
                                                        quantile.interval.my(y),
                                                        " ",
                                                        as.character(round(mean(y, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(y, na.rm = TRUE), 2))
                                      ),
                                      n.3      = n.z,
                                      group.3  = paste0( as.character(median.my(z)),
                                                                           " ",
                                                        quantile.interval.my(z),
                                                        " ",
                                                        as.character(round(mean(z, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(z, na.rm = TRUE), 2))
                                     ),
                                     diff     = "-",
                                     p.value  = "-",
                                     method      = "WARNING: ALL Na's values in 1st group"
                )
            
            }
            if(
                (all(is.na(x)) == FALSE) & 
                (all(is.na(y)) == TRUE)  &
                (all(is.na(z)) == FALSE)
            )
            {
                res.df = data.frame(  n.1      = n.x,
                                      group.1  = paste0(as.character(median.my(x)),
                                                        " ",
                                                        quantile.interval.my(x),
                                                        " ",
                                                        as.character(round(mean(x, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(x, na.rm = TRUE), 2))
                                      ),
                                      n.2      = n.y,
                                      group.2  = "NA",
                                      n.3      = n.z,
                                      group.3  = paste0(as.character(median.my(z)),
                                                        " ",
                                                        quantile.interval.my(z),
                                                        " ",
                                                        as.character(round(mean(z, na.rm = TRUE), 2)),
                                                        "?",
                                                        as.character(round(sd(z, na.rm = TRUE), 2))
                                      ),                  
                                      diff     = "-",
                                      p.value  = "-",
                                      method      = "WARNING: ALL Na's values in 2nd group"

                )
                
            }
            
            if(
                (all(is.na(x)) == TRUE) & 
                (all(is.na(y)) == TRUE) &
                (all(is.na(z)) == TRUE)
            )
            {
                res.df = data.frame( n.1      = n.x,
                                     group.1  = "NA",
                                     n.2      = n.y,
                                     group.2  = "NA",
                                     n.3      = n.z,
                                     group.3  = "NA",
                                     diff     = "-",
                                     p.value  = "-",
                                     method   = "WARNING: ALL Na's values in both group"
                )
                
            }
            
            rownames(res.df)[1] = colnames(data)[i.col]
            colnames(res.df)[1] = group.labels[1]
            colnames(res.df)[2] = group.labels[2]
            colnames(res.df)[2] = group.labels[3]
            result = rbind(result, res.df)
        }
        if(type.columns[i] == "bin")
        {
            # freq.matrix:
            # -----------------
            #        |  0  1
            # --------=--------
            # group1 |  7 33
            # group2 | 33 78
            #
            
            freq.matrix =  table(dt)
            nrow.f.m = nrow(freq.matrix)
            ncol.f.m = ncol(freq.matrix)
            
            if (dbg == 1) cat("\nfreq matrix         : ", freq.matrix)
            if (dbg == 1) cat("\ncolnames freq matrix: ", colnames(freq.matrix))
            if (dbg == 1) cat("\nrownames freq matrix: ", length(rownames(freq.matrix)),"\n")
            
            prcnt.1     = "NULL"
            prcnt.2     = "NULL"
            prcnt.3     = "NULL"
            res.p.value = "NULL"
            m.name      = "NULL"
            res.diff    = "NULL"
            
            
            
            if(
                (ncol.f.m == 2) & 
                (nrow.f.m == 2) &
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE) &
                (all(is.na(z)) == FALSE)
            ) 
            {    
                res = fisher.test(freq.matrix)
                
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                prcnt.3 = bin.my(z, digits=0)
                
                res.p.value = p_value_formatted(res$p.value)
                
                res.diff = ifelse(freq.matrix[1,2] == 0,
                                  "-",
                                  paste0( round(res$estimate,1)," [",
                                          round(res$conf.int[1],1),
                                          "; ",
                                          round(res$conf.int[2],1),
                                          "; ",
                                          round(res$conf.int[3],1),
                                          "]"
                                  )
                )
                m.name = "Fisher test"
            } 
            if (
                (
                    (length(colnames(freq.matrix)) == 1) & 
                    (length(rownames(freq.matrix)) == 2) &
                    (colnames(freq.matrix)[1] == 0)&
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE) &
                    (all(is.na(z)) == FALSE)
                )
                |
                (
                    (length(colnames(freq.matrix)) == 1) & 
                    (length(rownames(freq.matrix)) == 2) &
                    (colnames(freq.matrix)[1] == 1) &
                    (all(is.na(x)) == FALSE) &
                    (all(is.na(y)) == FALSE) &
                    (all(is.na(z)) == FALSE)
                )
                
            )
            {
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                prcnt.3 = bin.my(z, digits=0)
                res.p.value = ">0.999"
                res.diff    = "-"
                m.name      = "WARNING: equal cases in all groups"
            }
            
            if(
                (all(is.na(x)) == TRUE) 
                |
                (all(is.na(y)) == TRUE)
                |
                (all(is.na(z)) == TRUE)
            )
            {
                prcnt.1 = bin.my(x, digits=0)
                prcnt.2 = bin.my(y, digits=0)
                prcnt.3 = bin.my(z, digits=0)
                res.p.value = " - "
                res.diff    = " - "
                m.name      = "WARNING: All NA's !"
            }
            
            
            res.df = data.frame( n.1      = n.x,
                                 group.1  = prcnt.1,
                                 n.2      = n.y,
                                 group.2  = prcnt.2,
                                 n.3      = n.z,
                                 group.3  = prcnt.3,
                                 diff     = res.diff,
                                 p.value  = as.character(res.p.value),
                                 method   = m.name
            )
            rownames(res.df)[1] = colnames(data)[i.col]
            colnames(res.df)[1] = group.labels[1]
            colnames(res.df)[2] = group.labels[2]
            colnames(res.df)[3] = group.labels[3]
            result = rbind(result, res.df)
        }
        if(type.columns[i] == "cat")
        {
            
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == FALSE) &
                (all(is.na(z)) == FALSE)
            )
            {
                n.matrix    = table(dt)
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                # the code below concatenates rows of n.matrix and prop.matrix in the following format:
                # (this is example of group.1 cell content in data.frame)
                #  ----------------
                # | 0 - 35 (31.2%) |
                # | 1 - 76 (67.9%) |
                # | 2 - 0 (0%)     |
                # | 3 - 1 (0.9%)   |
                #  ----------------
                # where 0, 1, 2, 3 - category names/IDs
                # 35, 76, 0, 1 number of peoples in each category
                # % - how much peoples (in percentages) in each category (100% - total number of peoples in group = 35+76+0+1 = 112)
                res1 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[1,]), " (", as.numeric(prop.matrix[1,]), "%)"), collapse = "\n")
                res2 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[2,]), " (", as.numeric(prop.matrix[2,]), "%)"), collapse = "\n")
                res3 = paste(paste0(colnames(n.matrix), " - ", as.numeric(n.matrix[3,]), " (", as.numeric(prop.matrix[3,]), "%)"), collapse = "\n")
                
                if (
                    dim(n.matrix)[2] > 1
                )
                {
                    res = fisher.test(n.matrix)
                    res.p.value = as.character(p_value_formatted(res$p.value))
                    res.method  = "Fisher test"
                }
                if (
                    dim(n.matrix)[2] == 1
                )
                {
                    res.p.value = " - "
                    res.method  = "WARNING: only one category!"
                }
                
            }
            
            if(
                (all(is.na(x)) == TRUE) &
                (all(is.na(y)) == FALSE) &
                (all(is.na(z)) == FALSE)
            )
            {
                n.matrix    = table(as.data.frame(cbind(y,z)))
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                res1 = " - "
                res2 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                res3 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[2]), " (", as.numeric(prop.matrix[2]), "%)"), collapse = "\n")
                res.p.value = " - "
                res.method  = "WARNING: NA's values only in 1st group!"
                
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
            }
            if(
                (all(is.na(x)) == FALSE) &
                (all(is.na(y)) == TRUE) &
                (all(is.na(z)) == FALSE)
            )
            {
                n.matrix    = table(as.data.frame(cbind(x,z)))
                prop.matrix = round((prop.table(n.matrix, 1) * 100), 1)
                
                res1 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[1]), " (", as.numeric(prop.matrix[1]), "%)"), collapse = "\n")
                res2 = " - "
                res3 = paste(paste0(names(n.matrix), " - ", as.numeric(n.matrix[2]), " (", as.numeric(prop.matrix[2]), "%)"), collapse = "\n")
                res.p.value = " - "
                res.method  = "WARNING: NA's values only in 2nd group!"
                
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
            }
            if(
                (all(is.na(x)) == TRUE) &
                (all(is.na(y)) == TRUE) &
                (all(is.na(z)) == TRUE)
            )
            {
                res1 = " - "
                res2 = " - "
                res3 = " - "
                res.p.value = " - "
                res.method  = "WARNING: NA's values in all groups"
                
                n.matrix = NULL
                rownames(n.matrix)[1] = group.labels[1] # for right joining to result
                
            }
            
            res.df = data.frame( n.1      = n.x,
                                 group.1  = res1,
                                 n.2      = n.y,
                                 group.2  = res2,
                                 n.3      = n.z,
                                 group.3  = res3,
                                 diff     = " ",
                                 p.value  = res.p.value,
                                 method   = res.method
            )
            rownames(res.df)[1] = colnames(data)[i.col]
            
            # n.matrix rownames are ordered in the same way as initial data, not order we set for group.labels !!!
            if (group.labels[1] == rownames(n.matrix)[1]) {
                colnames(res.df)[1] = group.labels[1]
                colnames(res.df)[2] = group.labels[2]
                colnames(res.df)[3] = group.labels[3]
            } else {
                colnames(res.df)[1] = group.labels[3]
                colnames(res.df)[2] = group.labels[2]
                colnames(res.df)[3] = group.labels[1]
            }
            
            result = rbind(result, res.df)
        }
        
    }
    
    path.data = paste0(PATH, FILENAME,".csv")
    result %>% write.csv2(file=path.data)
    return(result)
}

# returns number of non NA's in vector:
# #non NA's(% non NA's)
na.count.string = function(x)
{
    return(
        paste0(as.character(sum(!is.na(x))),
               " (",
               as.character(round(sum(!is.na(x))/length(x)*100,0)),
               "%)"
        )
    )
}

# returns formatted value rounded to 3 digits:
# "0.001*"  if value < 0.001
# "0.xxx*"  if value < 0.05
# "0.xxx"   if 0.05 <= value < 1
# "1"       if value = 1
p_value_formatted = function(p_value)
{
    # number of digits for round
    nod = 3
    
    suffix_char = ""
    if (p_value < 0.001) {
        p_val = "<0.001"
    } else {
        p_val = round(p_value, nod)
    }
    
    if (p_value < 0.05){
        p_val = paste0(p_val, "*")
    }
    
    if (p_value == 1.0){
        p_val = ">0.999"
    }
    
    return (p_val)
}

#' Title
#' parse data.frame into two column and remove NA's  
#' first column contain group labels
#' second column contain values for comparing  
#'
#' @param dt            data frame for parsing
#' @param groups.column column contained group labels
#' @param values.column values column
#'
#' @return data frame
#' @export dpyr
#'
#' @examples  parse.data.frame(mtcars, 2, c(4,6), 3)
#' 
parse.data.frame = function(
    dt = NULL,
    groups.column = NULL,
    group.labels  = NULL,
    values.column = NULL
)
{
    require(dplyr)
    dt %>% 
        select(c(groups.column,values.column)) %>% 
        filter(dt[[groups.column]] %in% group.labels) %>% 
        na.omit() %>%
        return ()
}

parse.data.frame.with.na = function(
    dt = NULL,
    groups.column = NULL,
    group.labels  = NULL,
    values.column = NULL
)
{
    require(dplyr)
    dt %>% 
        select(c(groups.column,values.column)) %>% 
        filter(dt[[groups.column]] %in% group.labels) %>% 
        #na.omit() %>%
        return ()
}


#' Title
#'
#' @param id.group     id columns in data frame whith groups labels
#' @param group.labels vector of group labels
#' @param columns.list list containing id's and types: "num","bin" or "cat" of comparing columns 
#' @param data         data frame 
#' @param PATH         path to write result
#' @param FILENAME     filename for result
#'
#' @return
#' @export
#'
#' @examples
cmp.2group.all.vars = function(id.group,
                               group.labels = NULL,
                               columns.list = NULL,
                               data, 
                               PATH = NULL, 
                               FILENAME = NULL, 
                               result = NULL
)
{
    x = unlist(columns.list, recursive = TRUE)
    x.length = length(x)
    id.columns   = x[seq(1,x.length,2)]
    type.columns = x[seq(2,x.length,2)]
    id.columns   = as.integer(id.columns)
    type.columns = as.character(type.columns)
    
    res           = cmp.2group.all(id.group     = id.group,
                                   group.labels = group.labels,
                                   id.columns   = id.columns,
                                   type.columns = type.columns,
                                   data         = data, 
                                   PATH         = PATH, 
                                   FILENAME     = FILENAME, 
                                   result       = NULL
    )
    return (res)
}

calc.NA = function(dat = NULL, path.file = NULL)
{
    require(dplyr)
    res = summarise_if(dat,is.numeric,funs(sum((is.na(.)))))
    res = t(res)
    res = cbind(res,res)
    res[,2] = paste0(as.character(round((res[,2]/length(dat[[1]]))*100,0)), "%")
    res[,3] = round((res[,2]/length(dat[[1]]))*100,0)
    colnames(res) = c("?????????? ?????????","% ?????????","% ?????????")
    res %>% write.csv2(file=path.file)
    res = as.data.frame(res)
    res[,1] = as.integer(res[,1])
    res[,3] = as.integer(res[,3])
    return(res)
}


#########################3
summ.bin.date = function(
    dt       = NULL, # dt.diam
    offset   = 0,    # 2
    range    = 0,    # 1:19
    date.col = 0,    # 19
    stages   = 0,     # 5 length(offsets)
    format.date = "%d.%m.%Y"
)
{
    # remove id of date column
    k.range = range[!range %in% date.col] 
    
    k.range = k.range + offset
    
    for ( k in k.range )
    {
        #add summary columns
        bin.name             = paste0("total.bin ",  colnames(dt)[k])
        date.name            = paste0("first.date ", colnames(dt)[k])
        dt[ ,bin.name ]      = NA
        dt[ ,date.name]      = as.Date(dt[ ,offset+date.col],format.date)
        
        len.rows             = dim(dt)[1]
        
        for(id.row in 1:len.rows)  
        {
            #to know there was an event or not
            event = 0
            
            for(j in (1:stages - 1))  
            {
                offset.stage = j*length(range)
                id.col = k + offset.stage
                
                # current stage date
                cur.date = as.Date(dt[id.row, offset+date.col+offset.stage],format.date)
                
                if(event == 0)
                {
                    
                    if( 
                        ( !is.na(dt[id.row,id.col]) ) 
                        & 
                        ( dt[id.row,id.col] == 1    )
                    )
                    {
                        dt[id.row, bin.name]       = 1
                        dt.diam[id.row, date.name] = cur.date
                        event                      = 1 
                    } else
                    {
                        if( 
                            ( !is.na(cur.date)          )
                            &
                            ( !is.na(dt[id.row,id.col]) )
                            &
                            ( dt[id.row,id.col] == 0    )
                        )
                        {
                            dt[id.row, bin.name]  = 0
                            dt[id.row, date.name] = cur.date
                        }
                    }
                    
                }
            }
        }
    }
    return (dt)
}

###################
