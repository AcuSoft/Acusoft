$(document).ready(function(){
    $("#Paneles_Form").submit(function(e){
            e.preventDefault();
            $.ajax({
                    url: $(this).attr('action'),
                    type: $(this).attr('method'),
                    data: {
                            'units':$('#units').val(),
                            'f': $('#f').val(),
                            'q': $('#q').val(),

                            'db': $('#db').val(),
                            'dh': $('#dh').val(),
                            'dm': $('#dm').val(),

                            'pb': $('#pb').val(),
                            'ph': $('#ph').val(),    
                            'pg': $('#pg').val(),
                            'pd': $('#pd').val(),
                            'pp': $('#pp').val(),

                            'tipo':$('#tipo option:selected').val(),
                    },

                    success: function(datos){
                        console.log(datos) 
                        
                        const grafico_panel = document.getElementById('grafico_panel');
                        const orificio = document.getElementsByClassName('orificio');
                        grafico_panel.innerHTML = "";

                        if($("#tipo").val() == "Difragmático"){
                                grafico_panel.innerHTML = "";
                                grafico_panel.style.padding = "0% 0% 0% 0%";
                                $("#td_dd" ).html("<p>" + datos.dd + " kg/m<sup>2</sup></p>");
                                $("#td_de" ).html("<p>" + datos.de + " cm</p>");

                                if (datos.db>=datos.dh){
                                        console.log("base mayor que altura");
                                        proporcion_panel = datos.dh/datos.db;
                                        console.log(proporcion_panel);
                                        document.getElementById('grafico_panel').style.width = 100 + "%";
                                        document.getElementById('grafico_panel').style.height = proporcion_panel*100 + "%";
        
                                }
        
                                if (datos.db<datos.dh){
                                        console.log("altura mayor que base");
                                        proporcion = datos.db/datos.dh;
                                        console.log(proporcion);
                                        document.getElementById('grafico_panel').style.height = 100 + "%";
                                        document.getElementById('grafico_panel').style.width = proporcion_panel*100 + "%";

                                }
                        }
                        if($("#tipo").val() == "Perforado"){
                                grafico_panel.innerHTML = "";
                                $("#td_pnb" ).html("<p>" + datos.pnb + "</p>");
                                $("#td_pnh" ).html("<p>" + datos.pnh + "</p>");
                                $("#td_pn" ).html("<p>" + datos.pn + "</p>");
                                $("#td_ps" ).html("<p>" + datos.ps + " cm</p>");
                                $("#td_pe" ).html("<p>" + datos.pe + " cm</p>");
                                $("#td_bordebase").html("<p>" + datos.borde_base + " cm</p>");
                                $("#td_bordealtura").html("<p>" + datos.borde_altura + " cm</p>");

                                for (k=0;k<datos.pnh;k++){
                                        for (i=1;i <= datos.pnb;i++){
                                                grafico_panel.insertAdjacentHTML( 'beforeend', "<div class = 'orificio'></div>" );
                                                if (i == datos.pnb && k != datos.pnh-1){
                                                        grafico_panel.insertAdjacentHTML( 'beforeend', "<div class='break'></div>" );
                                                }
                                        }
                                }
        
                                $(".orificio").width(datos.pd*2.54 + "px");
                                $(".orificio").height(datos.pd*2.54 + "px");
        
                                if (datos.pb>=datos.ph){
                                        console.log("base mayor que altura");
                                        proporcion_panel = datos.ph/datos.pb;
                                        console.log(proporcion_panel);
                                        document.getElementById('grafico_panel').style.width = 100 + "%";
                                        document.getElementById('grafico_panel').style.height = proporcion_panel*100 + "%";
        
                                        porcentaje_bordealtura = datos.borde_altura*100/datos.pb;
                                        porcentaje_bordebase = datos.borde_base*proporcion_panel*100/datos.ph;
        
                                        grafico_panel.style.padding = porcentaje_bordebase + "% " + porcentaje_bordealtura + "% " + porcentaje_bordebase + "% " + porcentaje_bordealtura + "%";
        
                                        porcentaje_anchoorificio = datos.pd*100*2.54/datos.pb;
                                        porcentaje_altoorificio = datos.pd*100*2.54/datos.ph;                                
        
                                        $(".orificio").width(porcentaje_anchoorificio + "%");
                                        $(".orificio").height( porcentaje_altoorificio + "%");
        
                                }
        
                                if (datos.pb<datos.ph){
                                        console.log("altura mayor que base");
                                        proporcion_panel = datos.pb/datos.ph;
                                        console.log(proporcion_panel);
                                        document.getElementById('grafico_panel').style.height = 100 + "%";
                                        document.getElementById('grafico_panel').style.width = proporcion_panel*100 + "%";
        
                                        porcentaje_bordealtura = datos.borde_altura*proporcion_panel*100/datos.pb;
                                        porcentaje_bordebase = datos.borde_base*100/datos.ph;
        
                                        grafico_panel.style.padding = porcentaje_bordebase + "% " + porcentaje_bordealtura + "% " + porcentaje_bordebase + "% " + porcentaje_bordealtura + "%";
        
                                        porcentaje_anchoorificio = datos.pd*100*2.54/datos.pb;
                                        porcentaje_altoorificio = datos.pd*100*2.54/datos.ph;                                
        
                                        $(".orificio").width(porcentaje_anchoorificio + "%");
                                        $(".orificio").height( porcentaje_altoorificio + "%");
        
                                }
                        }

                                                  
                        
                        $("#td_bw" ).html("<p>" + datos.bw + " hz</p>");
                        $("#td_fh" ).html("<p>" + datos.fh + " hz</p>");
                        $("#td_fl" ).html("<p>" + datos.fl + " hz</p>");
                        $("#td_RT60" ).html("<p>" + datos.RT60 + " s</p>");


                       
                        
                    },
                    error: function(){
                            console.log("error");
                    }      
            })
    })
})



const Tipo = document.getElementById('tipo');

document.getElementById('R2_D').style.display = "none";
document.getElementById('R3_D').style.display = "none";
document.getElementById('R2_P').style.display = "none";
document.getElementById('R3_P').style.display = "none";

Tipo.addEventListener ('input', () => {

        $('#f').val("");

        $('#db').val("");
        $('#dh').val("");
        $('#dm').val("");

        $('#pb').val("");
        $('#ph').val("");    
        $('#pg').val("");
        $('#pd').val("");
        $('#pp').val("");

        tipo_panel = tipo.value;
        if (tipo_panel == "Difragmático"){
                console.log("Difragmático");
                document.getElementById('R2_D').style.display = "block";
                document.getElementById('R3_D').style.display = "block";
                document.getElementById('R2_P').style.display = "none";
                document.getElementById('R3_P').style.display = "none";
        }
        if (tipo_panel == "Perforado"){
                console.log("Perforado");
                document.getElementById('R2_P').style.display = "block";
                document.getElementById('R3_P').style.display = "block";
                document.getElementById('R2_D').style.display = "none";
                document.getElementById('R3_D').style.display = "none";
        }
        if (tipo_panel == ""){
                document.getElementById('R2_D').style.display = "none";
                document.getElementById('R3_D').style.display = "none";
                document.getElementById('R2_P').style.display = "none";
                document.getElementById('R3_P').style.display = "none";
                console.log("");
                
        }
              
})

var min = 0.20,
max = 100;

function slide() {
    var slider = document.getElementById("slider");
    var position = slider.value;
    var minp = slider.min;
    var maxp = slider.max;
    var minv = Math.log(min);
    var maxv = Math.log(max);
    var scale = (maxv-minv) / (maxp-minp);
    var value = Math.exp(minv + scale*(position-minp));
    var label = document.getElementById("label");
    label.innerHTML = Math.round(value);
    $('#q').val(value.toFixed(2));
}