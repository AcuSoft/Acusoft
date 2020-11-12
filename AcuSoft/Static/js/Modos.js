$(document).ready(function(){
    $("#Modos_Form").submit(function(e){
            e.preventDefault();
            $.ajax({
                url: $(this).attr('action'),
                type: $(this).attr('method'),
                data: {
                        'x': $('#x').val(),
                        'y': $('#y').val(),
                        'z': $('#z').val(),
                        'RT60':$('#RT60').val()
                    },

                    success: function(datos){
                        console.log(datos)

                        for (var i = 0; i<16;i++){
                            $("#td_xmA"+i).html("<p>" + datos.conteo[0][i-1] + " </p>");
                        }
                        for (var i = 0; i<16;i++){
                            $("#td_ymA"+i).html("<p>" + datos.conteo[1][i-1] + " </p>");
                        }
                        for (var i = 0; i<16;i++){
                            $("#td_zmA"+i).html("<p>" + datos.conteo[2][i-1] + " </p>");
                        }
                        for (var i = 0; i<16;i++){
                            $("#td_smA"+i).html("<p>" + datos.conteo[3][i-1] + " </p>");
                        }
                        
                        for (var i = 0; i<20;i++){
                            if (datos.axial[i][0] < 562){
                                $("#td_xfA"+i).html("<p>" + datos.axial[i][0] + " hz</p>");
                            }
                            
                        }
                        for (var i = 0; i<20;i++){
                            if (datos.axial[i][1] < 562){
                                $("#td_yfA"+i).html("<p>" + datos.axial[i][1] + " hz</p>");
                            }
                            
                        }
                        for (var i = 0; i<20;i++){
                            if (datos.axial[i][2] < 562){
                                $("#td_zfA"+i).html("<p>" + datos.axial[i][2] + " hz</p>");
                            }
                            
                        }
                        
                        var xp = datos.x/datos.x;
                        var yp = datos.y/datos.x;
                        var zp = datos.z/datos.x;

                        var grafica_conteo = datos.conteo[3]

                        $("#td_f1A").html("<p>" + 16 + " hz</p>");
                        $("#td_f1B").html("<p>" + datos.f1 + " hz</p>");

                        $("#td_f2A").html("<p>" + datos.f1 + " hz</p>");
                        $("#td_f2B").html("<p>" + datos.f2 + " hz</p>");

                        $("#td_f3A").html("<p>" + datos.f2 + " hz</p>");
                        $("#td_f3B").html("<p>" + datos.f3 + " hz</p>");

                        $("#td_f4A").html("<p>" + datos.f3 + " hz</p>");
                        $("#td_f4B").html("<p>" + 20000 + " hz</p>");

                        $("#td_sf").html("<p>" + datos.sf + " hz</p>");
                        $("#td_bw").html("<p>" + datos.bw + " hz</p>");

                        $("#td_xp").html("<p>" + xp.toFixed(2) + " : </p>");
                        $("#td_yp").html("<p>" + yp.toFixed(2) + " : </p>");
                        $("#td_zp").html("<p>" + zp.toFixed(2) + "</p>");

                        
                        var ctx=document.getElementById("myChart").getContext("2d");
                        var myChart = new Chart(ctx,{
                            type:"bar",
                            data:{
                                labels:[20,25,31,40,50,63,80,100,125,160,200,250,315,400,500],
                                datasets:[{
                                    data:grafica_conteo,
                                    backgroundColor:[
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                        'rgb(255,192,0)',
                                    ]
                                }]
                            },
                            options:{
                                scales:{
                                    yAxes:[{
                                        ticks:{
                                            beginAtZero:true
                                        }
                                    }]
                                }
                            }
                        })

                    },
                    error: function(){
                            console.log("error");
                }      
            })
    })
})


// const conteo = document.getElementById('Conteo');
// const GraficaModos = document.getElementById('Grafica');

// document.getElementById('L3').style.display = "none";
// document.getElementById('tabla2').style.display = "none";

// conteo.addEventListener('click',()=>{
//     document.getElementById('tabla1').style.display = "block";
//     document.getElementById('tabla2').style.display = "none";
//     document.getElementById('L3').style.display = "none";
// })

// GraficaModos.addEventListener('click',()=>{
//     document.getElementById('L3').style.display = "block";
//     document.getElementById('tabla2').style.display = "block";
//     document.getElementById('tabla1').style.display = "none";
// })


const Tipo_Cuarto = document.getElementById('Tipo_Cuarto');
const x_modo = document.getElementById('x');
const y_modo = document.getElementById('y');
const z_modo = document.getElementById('z');

Tipo_Cuarto.addEventListener('input',()=>{
    if (Tipo_Cuarto.value == "Voz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.343+(0.305*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }
    if (Tipo_Cuarto.value == "Jazz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.236+(0.319*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Orquesta"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.1034+(0.332*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Religiosa"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.4878+(0.314*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Personalizado"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val("");
        $('#RT60').prop("disabled", false );
        console.log(v);
    }

    
})

x_modo.addEventListener('input',()=>{
    if (Tipo_Cuarto.value == "Voz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.343+(0.305*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }
    if (Tipo_Cuarto.value == "Jazz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.236+(0.319*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Orquesta"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.1034+(0.332*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Religiosa"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.4878+(0.314*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Personalizado"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val("");
        $('#RT60').prop("disabled", false );
        console.log(v);
    }
})

y_modo.addEventListener('input',()=>{
    if (Tipo_Cuarto.value == "Voz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.343+(0.305*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }
    if (Tipo_Cuarto.value == "Jazz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.236+(0.319*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Orquesta"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.1034+(0.332*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Religiosa"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.4878+(0.314*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Personalizado"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val("");
        $('#RT60').prop("disabled", false );
        console.log(v);
    }  
})

z_modo.addEventListener('input',()=>{
    if (Tipo_Cuarto.value == "Voz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.343+(0.305*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }
    if (Tipo_Cuarto.value == "Jazz"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((-0.236+(0.319*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Orquesta"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.1034+(0.332*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Religiosa"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val((0.4878+(0.314*Math.log10(v))).toFixed(3));
        $('#RT60').prop("disabled", true );
        console.log(v);
    }

    if (Tipo_Cuarto.value == "Personalizado"){
        var v = $('#x').val()*$('#y').val()*$('#z').val();
        $('#RT60').val("");
        $('#RT60').prop("disabled", false );
        console.log(v);
    }   
})