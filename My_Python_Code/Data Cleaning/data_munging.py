import pandas as pdimport numpy as np###Read in datacen1900 = pd.read_csv('data/pe-11-1900.csv')cen1910 = pd.read_csv('data/pe-11-1910.csv')cen1920 = pd.read_csv('data/pe-11-1920.csv')cen1930 = pd.read_csv('data/pe-11-1930.csv')cen1940 = pd.read_csv('data/pe-11-1940.csv')cen1950 = pd.read_csv('data/pe-11-1950.csv')cen1960 = pd.read_csv('data/pe-11-1960.csv')cen1970 = pd.read_csv('data/pe-11-1970.csv')################################   Part 1: 1900-1970 Munging   ############################################1. Rename column namescen1900.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]######2.Remove unnecessary rowscen1900 = cen1900.drop(index=[0,1,2,3,4,5,6,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97])######3. Keep certain columnscen1900 = cen1900[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]#####4. Regroup rows to every 5 years (total 16 age group)#(1) Remove "+"cen1900.iat[75,0] = 75#(2) Remove all ","names = [["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]for name in cen1900[["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]:    cen1900[name] = cen1900[name].str.replace(',','')#(3) change datatype to numericconvert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1900 = cen1900.astype(convert_dict)#(4) Groupby every 5 rowsclean1900 = cen1900.groupby(cen1900.Age_group // 5).sum()#(5) Make age_group aligned to the 2000 dataclean1900['Age_group'] = pd.Series(range(1,17))#(6) Reset index#clean1900 = clean1900.reset_index(drop = True, inplace =True)######6. Add year columnclean1900.insert(clean1900.shape[1],'Year',1900)print("After cleaning \n", clean1900)######4. Repeat the procedure above##1910##cen1910.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]cen1910 = cen1910.drop(index=[0,1,2,3,4,5,6,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97])cen1910 = cen1910[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]cen1910.iat[75,0] = 75for name in cen1910[["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]:    cen1910[name] = cen1910[name].str.replace(',','')convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1910 = cen1910.astype(convert_dict)clean1910 = cen1910.groupby(cen1910.Age_group // 5).sum()clean1910['Age_group'] = pd.Series(range(1,17))clean1910.insert(clean1910.shape[1],'Year',1910)print("After cleaning \n", clean1910.head(20))##1920##cen1920.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]cen1920 = cen1920.drop(index=[0,1,2,3,4,5,6,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97])cen1920 = cen1920[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]cen1920.iat[75,0] = 75for name in cen1920[["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]:    cen1920[name] = cen1920[name].str.replace(',','')convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1920 = cen1920.astype(convert_dict)clean1920 = cen1920.groupby(cen1920.Age_group // 5).sum()clean1920['Age_group'] = pd.Series(range(1,17))clean1920.insert(clean1920.shape[1],'Year',1920)print("After cleaning \n", clean1920.head(20))##1930##cen1930.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]cen1930 = cen1930.drop(index=[0,1,2,3,4,5,6,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97])cen1930 = cen1930[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]cen1930.iat[75,0] = 75for name in cen1930[["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]:    cen1930[name] = cen1930[name].str.replace(',','')convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1930 = cen1930.astype(convert_dict)clean1930 = cen1930.groupby(cen1930.Age_group // 5).sum()clean1930['Age_group'] = pd.Series(range(1,17))clean1930.insert(clean1930.shape[1],'Year',1930)print("After cleaning \n", clean1930.head(20))##1940##cen1940.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]cen1940 = cen1940.drop(index=[0,1,2,3,4,5,6,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107])cen1940 = cen1940[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]cen1940.iat[85,0] = 85for name in cen1940[["WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]:    cen1940[name] = cen1940[name].str.replace(',','')convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1940 = cen1940.astype(convert_dict)#sum 75+ to 75sum75 = cen1940.iloc[75:86].sum()cen1940 = cen1940.append(sum75,ignore_index=True)#Drop 75 - 85, change Age_group 75cen1940 = cen1940.drop(cen1940[(cen1940["Age_group"] > 74) & (cen1940["Age_group"] < 100)].index)cen1940#change Age_group to 75cen1940.iat[75,0] = 75#Groupbyclean1940 = cen1940.groupby(cen1940.Age_group // 5).sum()clean1940['Age_group'] = pd.Series(range(1,17))clean1940.insert(clean1940.shape[1],'Year',1940)print("After cleaning \n", clean1940.head(20))##1950##cen1950.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "NONWHT_TOT", "NONWHT_male", "NONWHT_female" ]cen1950 = cen1950.drop(index=[0,1,2,3,4,5,6,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107])#Change data type cen1950.iat[85,0] = 85for name in cen1950[["WHT_male","WHT_female","NONWHT_TOT", "NONWHT_male", "NONWHT_female"]]:    cen1950[name] = cen1950[name].str.replace(',','')    cen1950[name] = cen1950[name].str.replace(' ','')   convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"NONWHT_male":int,"NONWHT_female":int}cen1950 = cen1950.astype(convert_dict)cen1950 = cen1950[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]#sum 75+ to 75sum75 = cen1950.iloc[75:86].sum()cen1950 = cen1950.append(sum75,ignore_index=True)cen1950#Drop 75 - 85, change Age_group 75cen1950 = cen1950.drop(cen1950[(cen1950["Age_group"] > 74) & (cen1950["Age_group"] < 100)].index)cen1950#change Age_group to 75cen1950.iat[75,0] = 75#Groupbyclean1950 = cen1950.groupby(cen1950.Age_group // 5).sum()clean1950['Age_group'] = pd.Series(range(1,17))clean1950.insert(clean1950.shape[1],'Year',1950)print("After cleaning \n", clean1950.head(20))##1960##cen1960.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "BLK_TOT", "BLK_male", "BLK_female","OTH_TOT","OTH_male","OTH_female"]cen1960 = cen1960.drop(index=[0,1,2,3,4,5,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106]) #Slightly change in rows#Change data type cen1960.iat[85,0] = 85for name in cen1960[["WHT_male","WHT_female","BLK_TOT", "BLK_male", "BLK_female","OTH_TOT","OTH_male","OTH_female"]]:    cen1960[name] = cen1960[name].str.replace(',','')    cen1960[name] = cen1960[name].str.replace(' ','')   convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"BLK_male":int,"BLK_female":int,"OTH_male":int,"OTH_female":int}cen1960 = cen1960.astype(convert_dict)cen1960["NONWHT_male"] = cen1960["BLK_male"]+cen1960["OTH_male"]cen1960["NONWHT_female"] = cen1960["BLK_female"]+cen1960["OTH_female"]cen1960 = cen1960[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]sum75 = cen1960.iloc[75:86].sum()cen1960 = cen1960.append(sum75,ignore_index=True)cen1960 = cen1960.drop(cen1960[(cen1960["Age_group"] > 74) & (cen1960["Age_group"] < 100)].index)cen1960.iat[75,0] = 75cen1960clean1960 = cen1960.groupby(cen1960.Age_group // 5).sum()clean1960['Age_group'] = pd.Series(range(1,17))clean1960.insert(clean1960.shape[1],'Year',1960)print("After cleaning \n", clean1960.head(20))##1970##cen1970.columns = ["Age_group", "TOT", "TOT_male", "TOT_female", "WHT_TOT","WHT_male", "WHT_female", "BLK_TOT", "BLK_male", "BLK_female","OTH_TOT","OTH_male","OTH_female"]cen1970 = cen1970.drop(index=[0,1,2,3,4,5,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106]) #Slightly change in rows#Change data type cen1970.iat[85,0] = 85for name in cen1970[["WHT_male","WHT_female","BLK_TOT", "BLK_male", "BLK_female","OTH_TOT","OTH_male","OTH_female"]]:    cen1970[name] = cen1970[name].str.replace(',','')    cen1970[name] = cen1970[name].str.replace(' ','')   convert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"BLK_male":int,"BLK_female":int,"OTH_male":int,"OTH_female":int}cen1970 = cen1970.astype(convert_dict)cen1970["NONWHT_male"] = cen1970["BLK_male"]+cen1970["OTH_male"]cen1970["NONWHT_female"] = cen1970["BLK_female"]+cen1970["OTH_female"]cen1970 = cen1970[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female"]]sum75 = cen1970.iloc[75:86].sum()cen1970 = cen1970.append(sum75,ignore_index=True)cen1970 = cen1970.drop(cen1970[(cen1970["Age_group"] > 74) & (cen1970["Age_group"] < 100)].index)cen1970.iat[75,0] = 75clean1970 = cen1970.groupby(cen1970.Age_group // 5).sum()clean1970['Age_group'] = pd.Series(range(1,17))clean1970.insert(clean1970.shape[1],'Year',1970)print("After cleaning \n", clean1970.head(20))################################   Part 2: 1980-1990 Munging   ######################################################################           A. 1980             ############################################1. Read in filecen1980 = pd.read_fwf('data/National-intercensal-data-1980.TXT',skipfooter=1)cen1980.head(20)cen1980.tail(20)#######Change column namescen1980.columns = ["Series","Year_age","TOT","TOT_M","TOT_F","WHT_male","WHT_female","BLK_M","BLK_F","AIEA_M","AIEA_F","ASIA_M","ASIA_F","HIS_M","HIS_F","WHT_NONHIS_M","WHT_NONHIS_F","BLK_NONHIS_M","BLK_NONHIS_F","AIEA_NONHIS_M","AIEA_NONHIS_F","AIEA_HIS_M","AIEA_HIS_F"]######2. Split Year_age to year and age.cen1980[['Year']] = cen1980['Year_age'].str[:3]cen1980[['Age_group']] = cen1980['Year_age'].str[-2:]######3. Select nessary rows and columns to clean dfcen1980 = cen1980.drop(columns=['Series','Year_age'])cen1980 = cen1980[cen1980['Year'] == '480']######4.Change str to numericconvert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"TOT_M":int,"TOT_F":int,"Year":int}cen1980 = cen1980.astype(convert_dict)######5. Calculate Non-whitecen1980['NONWHT_male'] = cen1980["TOT"] - cen1980["TOT_M"]cen1980['NONWHT_female'] = cen1980["TOT"] - cen1980["TOT_F"]cen1980 = cen1980[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female","Year"]]######6.sum 75+ to 75sum75 = cen1980.iloc[75:101].sum()cen1980 = cen1980.append(sum75,ignore_index=True)#Drop 75 - 100, change Age_group 75cen1980.iat[100,0] = 100cen1980 = cen1980.drop(cen1980[(cen1980["Age_group"] > 74) & (cen1980["Age_group"] < 101)].index)cen1980#change Age_group to 75cen1980.iat[75,0] = 75#Groupbyclean1980 = cen1980.groupby(cen1980.Age_group // 5).sum()clean1980['Age_group'] = pd.Series(range(1,17))clean1980['Year'] = pd.Series([1980]*17)print("After cleaning \n", clean1980.head(20))################################           B. 1980             ######################################cen1990 = pd.read_fwf('data/National-intercensal-data-1990.TXT',skipfooter=1)cen1990.head(25)#######Change column namescen1990.columns = ["Series","Year_age","TOT","TOT_M","TOT_F","WHT_male","WHT_female","BLK_M","BLK_F","AIEA_M","AIEA_F","ASIA_M","ASIA_F","HIS_M","HIS_F","WHT_NONHIS_M","WHT_NONHIS_F","BLK_NONHIS_M","BLK_NONHIS_F","AIEA_NONHIS_M","AIEA_NONHIS_F","AIEA_HIS_M","AIEA_HIS_F"]######2. Split Year_age to year and age.cen1990[['Year']] = cen1990['Year_age'].str[:3]cen1990[['Age_group']] = cen1990['Year_age'].str[-2:]######3. Select nessary rows and columns to clean dfcen1990 = cen1990.drop(columns=['Series','Year_age'])cen1990 = cen1990[cen1990['Year'] == '490']cen1990 = cen1990.drop(index=[407])###Remove the first sum row.######4.Change str to numericconvert_dict = {'Age_group':int,"WHT_male":int,"WHT_female":int,"TOT_M":int,"TOT_F":int,"Year":int}cen1990 = cen1990.astype(convert_dict)######5. Calculate Non-whitecen1990['NONWHT_male'] = cen1990["TOT_M"] - cen1990["WHT_male"]cen1990['NONWHT_female'] = cen1990["TOT_F"] - cen1990["WHT_female"]cen1990 = cen1990[["Age_group","WHT_male","WHT_female","NONWHT_male","NONWHT_female","Year"]]cen1990######6.sum 75+ to 75sum75 = cen1990.iloc[75:101].sum()cen1990 = cen1990.append(sum75,ignore_index=True)#Drop 75 - 100, change Age_group 75cen1990.iat[100,0] = 100cen1990 = cen1990.drop(cen1990[(cen1990["Age_group"] > 74) & (cen1990["Age_group"] < 101)].index)cen1990#change Age_group to 75cen1990.iat[75,0] = 75#Groupbyclean1990 = cen1990.groupby(cen1990.Age_group // 5).sum()clean1990['Age_group'] = pd.Series(range(1,17))clean1990['Year'] = pd.Series([1990]*17)print("After cleaning \n", clean1990.head(20)) ################################   Part 3: 2000-2010 Munging   ################################################           2000         ################cen2000 = pd.read_csv('data/National-intercensal-data-2000-2010.csv')cen2000.head(20)##### 1.Select month=4cen2000 = cen2000[(cen2000["month"] == 4) & (cen2000["year"] == 2000)]##### 2.Change data type to numericconvert_dict = {'AGEGRP':int,"WA_MALE":int,"WA_FEMALE":int,"TOT_MALE":int,"TOT_FEMALE":int,"year":int}cen2000 = cen2000.astype(convert_dict)##### 3.Calculate Non-whitecen2000['NONWHT_male'] = cen2000["TOT_MALE"] - cen2000["WA_MALE"]cen2000['NONWHT_female'] = cen2000["TOT_FEMALE"] - cen2000["WA_FEMALE"]cen2000.tail(25)##### 4.sum 75+ to 75sum75 = cen2000.iloc[16:22].sum()cen2000 = cen2000.append(sum75,ignore_index=True)cen2000# 5.Remove unuseful columnscen2000 = cen2000.drop(cen2000[((cen2000['AGEGRP']>15) & (cen2000['AGEGRP'] < 22)) | (cen2000["AGEGRP"] == 0)].index)cen2000.iat[15,2] = 16cen2000.iat[15,1] = 2000clean2000 = cen2000[["AGEGRP","WA_MALE","WA_FEMALE","NONWHT_male","NONWHT_female","year"]]# 6.Rename columnsclean2000 = clean2000.rename(columns={"AGEGRP":"Age_group", "WA_MALE":"WHT_male", "WA_FEMALE":"WHT_female", "year":"Year"})print("After cleaning \n", clean2000)################################           2010         ################cen2010 = pd.read_csv('data/National-intercensal-data-2000-2010.csv')cen2010.head(20)##### 1.Select month=4cen2010 = cen2010[(cen2010["month"] == 4) & (cen2010["year"] == 2010)]##### 2.Change data type to numericconvert_dict = {'AGEGRP':int,"WA_MALE":int,"WA_FEMALE":int,"TOT_MALE":int,"TOT_FEMALE":int,"year":int}cen2010 = cen2010.astype(convert_dict)##### 3.Calculate Non-whitecen2010['NONWHT_male'] = cen2010["TOT_MALE"] - cen2010["WA_MALE"]cen2010['NONWHT_female'] = cen2010["TOT_FEMALE"] - cen2010["WA_FEMALE"]cen2010.tail(25)##### 4.sum age group 75+ as 75sum75 = cen2010.iloc[16:22].sum()cen2010 = cen2010.append(sum75,ignore_index=True)# 5.Remove unuseful columnscen2010 = cen2010.drop(cen2010[((cen2010['AGEGRP']>15) & (cen2010['AGEGRP'] < 22)) | (cen2010["AGEGRP"] == 0)].index)cen2010.iat[15,2] = 16cen2010.iat[15,1] = 2010clean2010 = cen2010[["AGEGRP","WA_MALE","WA_FEMALE","NONWHT_male","NONWHT_female","year"]]# 6.Rename columnsclean2010 = clean2010.rename(columns={"AGEGRP":"Age_group", "WA_MALE":"WHT_male", "WA_FEMALE":"WHT_female", "year":"Year"})print("After cleaning \n", clean2010)################################   Part 4: Combine all dataframes together  ######################################cleandf = pd.concat([clean1900,clean1910,clean1920,clean1930,clean1940,clean1950,clean1960,clean1970,clean1980,clean1990,clean2000,clean2010],axis = 0)cleandf = cleandf.reset_index(drop = True)cleandf################################   Part 5: Tidy data  ######################################### Melt columnstidydf = cleandf.melt(id_vars = ["Age_group","Year"],                      var_name = "variable",                      value_name = "Population")###Split out race and sextidydf["Race"],tidydf["Sex"] = zip(*tidydf.variable.str.split("_"))###Select needed columnstidydf2 = tidydf[["Year","Age_group","Race","Sex","Population"]]tidydf2###Save to csvtidydf2.to_csv("part2-analytical-dataset.csv")